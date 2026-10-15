
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, EmailStr
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

if not MONGO_URI or not MONGO_DB:
    raise RuntimeError(
        "MONGO_URI and MONGO_DB must be configured in .env"
    )

IST = ZoneInfo("Asia/Kolkata")
UTC = timezone.utc

client = MongoClient(
    MONGO_URI,
    tz_aware=True,
    tzinfo=UTC,
    serverSelectionTimeoutMS=5000,
)

db = client[MONGO_DB]
employees = db["employees"]
attendance_logs = db["attendance_logs"]


# --------------------------------------------------
# 2. Request models
# --------------------------------------------------

class EmployeeCreateRequest(BaseModel):
    emp_code: str
    name: str
    email: EmailStr
    department: str
    joined_on: date
    shift_start: str = "09:30"
    shift_end: str = "18:30"


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: int = Field(ge=0)


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: int = Field(ge=0)


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def validate_shift_time(value: str) -> bool:
    if not isinstance(value, str):
        return False

    try:
        parsed = datetime.strptime(value, "%H:%M")
        return parsed.strftime("%H:%M") == value
    except ValueError:
        return False


def epoch_ms_to_datetime(value: int) -> datetime:
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    except (ValueError, OverflowError, OSError):
        raise HTTPException(
            status_code=422,
            detail="Invalid timestamp.",
        )


def ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)

    return value.astimezone(UTC)


def to_epoch_ms(value: datetime | None) -> int | None:
    if value is None:
        return None

    return int(ensure_aware_utc(value).timestamp() * 1000)


def get_attendance_date(value: datetime) -> str:
    return ensure_aware_utc(value).astimezone(IST).date().isoformat()


def round_half_up(value: float, places: int = 2) -> float:
    quantizer = Decimal("1").scaleb(-places)

    return float(
        Decimal(str(value)).quantize(
            quantizer,
            rounding=ROUND_HALF_UP,
        )
    )


def calculate_late_minutes(
    punched_at: datetime,
    shift_start: str,
) -> int:
    if not validate_shift_time(shift_start):
        raise ValueError("Invalid shift start time.")

    local_time = ensure_aware_utc(punched_at).astimezone(IST)

    scheduled_start = datetime.combine(
        local_time.date(),
        datetime.strptime(shift_start, "%H:%M").time(),
        tzinfo=IST,
    )

    difference_seconds = (
        local_time - scheduled_start
    ).total_seconds()

    # Count only complete minutes of lateness.
    return max(0, int(difference_seconds // 60))


def calculate_shift_end(
    attendance_day: date,
    shift_start: str,
    shift_end: str,
) -> datetime:
    start_time = datetime.strptime(
        shift_start, "%H:%M"
    ).time()

    end_time = datetime.strptime(
        shift_end, "%H:%M"
    ).time()

    end_day = attendance_day

    # Supports overnight shifts, such as 22:00 to 06:00.
    if end_time <= start_time:
        end_day += timedelta(days=1)

    return datetime.combine(end_day, end_time, tzinfo=IST)


def attendance_response(record: dict) -> dict:
    return {
        "emp_code": record["emp_code"],
        "date": record["date"],
        "status": record.get("status", "PRESENT"),
        "punch_in": to_epoch_ms(record.get("punch_in")),
        "punch_out": to_epoch_ms(record.get("punch_out")),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes", 0),
        "overtime_minutes": record.get("overtime_minutes", 0),
        "half_day": record.get("half_day", False),
        "history": record.get("history", []),
    }


def employee_response(record: dict) -> dict:
    return {
        "emp_code": record["emp_code"],
        "name": record["name"],
        "email": record["email"],
        "department": record["department"],
        "joined_on": record["joined_on"],
        "shift_start": record.get("shift_start", "09:30"),
        "shift_end": record.get("shift_end", "18:30"),
    }


# --------------------------------------------------
# 4. Application setup
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client.admin.command("ping")
        yield
    finally:
        client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 5. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    try:
        client.admin.command("ping")

        return {
            "status": "healthy",
            "database": "connected",
        }

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB connection failed.",
        )


# --------------------------------------------------
# 6. Create employee
# --------------------------------------------------

@app.post("/employees", status_code=201)
def create_employee(payload: EmployeeCreateRequest):
    if (
        not validate_shift_time(payload.shift_start)
        or not validate_shift_time(payload.shift_end)
    ):
        raise HTTPException(
            status_code=422,
            detail="Shift times must use HH:MM format.",
        )

    employee = {
        "emp_code": payload.emp_code,
        "name": payload.name,
        "email": str(payload.email),
        "department": payload.department,
        "joined_on": payload.joined_on.isoformat(),
        "shift_start": payload.shift_start,
        "shift_end": payload.shift_end,
    }

    try:
        if employees.find_one({"emp_code": payload.emp_code}):
            raise HTTPException(
                status_code=409,
                detail="Employee code already exists.",
            )

        if employees.find_one({"email": str(payload.email)}):
            raise HTTPException(
                status_code=409,
                detail="Email already exists.",
            )

        employees.insert_one(employee)

        return employee_response(employee)

    except HTTPException:
        raise

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not create employee in MongoDB.",
        )


# --------------------------------------------------
# 7. List employees
# --------------------------------------------------

@app.get("/employees")
def list_employees(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    try:
        total = employees.count_documents({})

        records = (
            employees.find()
            .sort("emp_code", ASCENDING)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )

        return {
            "items": [
                employee_response(item)
                for item in records
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve employees from MongoDB.",
        )


# --------------------------------------------------
# 8. Punch in
# --------------------------------------------------

@app.post("/attendance/punch-in", status_code=201)
def punch_in(payload: PunchInRequest):
    try:
        employee = employees.find_one(
            {"emp_code": payload.emp_code}
        )

        if employee is None:
            raise HTTPException(
                status_code=404,
                detail="Employee not found.",
            )

        punched_at = epoch_ms_to_datetime(payload.punched_at)
        attendance_date = get_attendance_date(punched_at)

        existing = attendance_logs.find_one({
            "emp_code": payload.emp_code,
            "date": attendance_date,
        })

        if existing:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Attendance already exists for this "
                    "employee and date."
                ),
            )

        shift_start = employee.get("shift_start", "09:30")

        if not validate_shift_time(shift_start):
            raise HTTPException(
                status_code=500,
                detail="Employee has an invalid shift configuration.",
            )

        record = {
            "emp_code": payload.emp_code,
            "date": attendance_date,
            "status": "PRESENT",
            "punch_in": punched_at,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": calculate_late_minutes(
                punched_at,
                shift_start,
            ),
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        attendance_logs.insert_one(record)

        return attendance_response(record)

    except HTTPException:
        raise

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not save attendance in MongoDB.",
        )


# --------------------------------------------------
# 9. Punch out
# --------------------------------------------------

@app.post("/attendance/punch-out")
def punch_out(payload: PunchOutRequest):
    try:
        employee = employees.find_one(
            {"emp_code": payload.emp_code}
        )

        if employee is None:
            raise HTTPException(
                status_code=404,
                detail="Employee not found.",
            )

        punched_at = epoch_ms_to_datetime(payload.punched_at)

        # Find the latest unclosed attendance record.
        record = attendance_logs.find_one(
            {
                "emp_code": payload.emp_code,
                "punch_in": {"$lte": punched_at},
                "punch_out": None,
            },
            sort=[("punch_in", DESCENDING)],
        )

        if record is None:
            previous_record = attendance_logs.find_one(
                {
                    "emp_code": payload.emp_code,
                    "punch_in": {"$lte": punched_at},
                },
                sort=[("punch_in", DESCENDING)],
            )

            if previous_record:
                raise HTTPException(
                    status_code=409,
                    detail="Attendance has already been punched out.",
                )

            raise HTTPException(
                status_code=404,
                detail="No matching punch-in record found.",
            )

        punch_in_time = ensure_aware_utc(record["punch_in"])
        punched_at = ensure_aware_utc(punched_at)

        if punched_at <= punch_in_time:
            raise HTTPException(
                status_code=422,
                detail="Punch-out must be after punch-in.",
            )

        if punched_at > punch_in_time + timedelta(hours=24):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Punch-out cannot be more than "
                    "24 hours after punch-in."
                ),
            )

        shift_start = employee.get("shift_start", "09:30")
        shift_end = employee.get("shift_end", "18:30")

        if (
            not validate_shift_time(shift_start)
            or not validate_shift_time(shift_end)
        ):
            raise HTTPException(
                status_code=500,
                detail="Employee has an invalid shift configuration.",
            )

        attendance_day = date.fromisoformat(record["date"])

        scheduled_end = calculate_shift_end(
            attendance_day,
            shift_start,
            shift_end,
        )

        punch_in_local = punch_in_time.astimezone(IST)
        punch_out_local = punched_at.astimezone(IST)

        # Calculate actual elapsed working time.
        work_seconds = (
            punch_out_local - punch_in_local
        ).total_seconds()

        # Keep two decimal places for work hours.
        work_hours = round_half_up(
            work_seconds / 3600,
            places=2,
        )

        # Use exact elapsed time, not rounded work hours.
        half_day = work_seconds < (4.5 * 3600)

        # Count overtime only when the employee's work
        # period extends beyond the scheduled shift end.
        overtime_seconds = 0

        if (
            punch_in_local < scheduled_end
            and punch_out_local > scheduled_end
        ):
            overtime_seconds = (
                punch_out_local - scheduled_end
            ).total_seconds()

        # Overtime is recorded only if it reaches 30 minutes.
        overtime_minutes = (
            int(overtime_seconds // 60)
            if overtime_seconds >= 30 * 60
            else 0
        )

        result = attendance_logs.update_one(
            {
                "_id": record["_id"],
                "punch_out": None,
            },
            {
                "$set": {
                    "punch_out": punched_at,
                    "work_hours": work_hours,
                    "overtime_minutes": overtime_minutes,
                    "half_day": half_day,
                }
            },
        )

        if result.modified_count != 1:
            raise HTTPException(
                status_code=409,
                detail="Attendance was modified by another request.",
            )

        updated_record = attendance_logs.find_one(
            {"_id": record["_id"]}
        )

        if updated_record is None:
            raise HTTPException(
                status_code=500,
                detail="Could not retrieve updated attendance.",
            )

        return attendance_response(updated_record)

    except HTTPException:
        raise

    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid attendance date or shift time.",
        )

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not update attendance in MongoDB.",
        )


# --------------------------------------------------
# 10. List attendance
# --------------------------------------------------

@app.get("/attendance")
def list_attendance(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    emp_code: str | None = None,
):
    try:
        query = {}

        if emp_code:
            query["emp_code"] = emp_code

        total = attendance_logs.count_documents(query)

        records = (
            attendance_logs.find(query)
            .sort(
                [
                    ("date", DESCENDING),
                    ("emp_code", ASCENDING),
                ]
            )
            .skip((page - 1) * page_size)
            .limit(page_size)
        )

        return {
            "items": [
                attendance_response(item)
                for item in records
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve attendance from MongoDB.",
        )

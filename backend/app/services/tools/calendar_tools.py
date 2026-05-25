from __future__ import annotations

import json
import os
import re

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# Read from environment variables. Set these in your .env:
#
#   CALENDAR_BACKEND   google | caldav | local  (default: local)
#
#   --- Google Calendar ---
#   GOOGLE_CREDENTIALS_PATH   path to service account or OAuth JSON
#   GOOGLE_CALENDAR_ID        calendar ID (default: 'primary')
#
#   --- CalDAV ---
#   CALDAV_URL         CalDAV server URL
#   CALDAV_USER        username
#   CALDAV_PASSWORD    password
#
#   --- Local (JSON file, for testing) ---
#   LOCAL_CALENDAR_PATH   path to local JSON calendar file (default: calendar.json)
# =========================================================

def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# =========================================================
# SCHEMAS
# =========================================================

class ListEventsSchema(BaseModel):
    start_date: Optional[str] = Field(None, description="Start of range in ISO 8601 (e.g. '2024-11-01' or '2024-11-01T09:00:00'). Defaults to today.")
    end_date: Optional[str] = Field(None, description="End of range in ISO 8601. Defaults to 7 days from start.")
    calendar_id: Optional[str] = Field(None, description="Calendar ID to query. Uses default if not set.")
    max_results: int = Field(50, ge=1, le=250, description="Maximum number of events to return.")


class CreateEventSchema(BaseModel):
    title: str = Field(..., description="Event title / summary.")
    start: str = Field(..., description="Start datetime in ISO 8601 (e.g. '2024-11-04T14:00:00').")
    end: str = Field(..., description="End datetime in ISO 8601.")
    description: Optional[str] = Field(None, description="Event description / notes.")
    location: Optional[str] = Field(None, description="Physical location or video call link.")
    attendees: Optional[List[str]] = Field(None, description="List of attendee email addresses.")
    timezone: str = Field("UTC", description="Timezone for the event (e.g. 'America/New_York').")
    calendar_id: Optional[str] = Field(None, description="Target calendar ID.")
    recurrence: Optional[str] = Field(None, description="RRULE string for recurring events (e.g. 'RRULE:FREQ=WEEKLY;COUNT=4').")


class UpdateEventSchema(BaseModel):
    event_id: str = Field(..., description="ID of the event to update.")
    title: Optional[str] = Field(None, description="New event title.")
    start: Optional[str] = Field(None, description="New start datetime in ISO 8601.")
    end: Optional[str] = Field(None, description="New end datetime in ISO 8601.")
    description: Optional[str] = Field(None, description="New description.")
    location: Optional[str] = Field(None, description="New location.")
    attendees: Optional[List[str]] = Field(None, description="Replacement attendee list.")
    calendar_id: Optional[str] = Field(None, description="Calendar ID containing the event.")


class DeleteEventSchema(BaseModel):
    event_id: str = Field(..., description="ID of the event to delete.")
    calendar_id: Optional[str] = Field(None, description="Calendar ID containing the event.")


class CheckAvailabilitySchema(BaseModel):
    emails: List[str] = Field(..., description="List of email addresses to check availability for.")
    start: str = Field(..., description="Start of the window to check in ISO 8601.")
    end: str = Field(..., description="End of the window to check in ISO 8601.")
    calendar_id: Optional[str] = Field(None, description="Calendar ID to check against.")


# =========================================================
# LOCAL CALENDAR BACKEND (default / testing)
# Stores events as a JSON file. Swap out for Google / CalDAV
# by implementing the same interface below.
# =========================================================

class LocalCalendarBackend:
    """
    Simple JSON-file calendar backend for local testing.
    Each event is a dict matching a subset of the Google Calendar event schema.
    """

    def __init__(self, path: str = "calendar.json"):
        self.path = Path(path)
        self._ensure_file()

    def _ensure_file(self):
        if not self.path.exists():
            self.path.write_text(json.dumps({"events": []}, indent=2))

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]):
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _new_id(self, data: Dict[str, Any]) -> str:
        existing = {e["id"] for e in data["events"]}
        import uuid
        while True:
            candidate = uuid.uuid4().hex[:12]
            if candidate not in existing:
                return candidate

    def list_events(
        self,
        start: datetime,
        end: datetime,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        data = self._load()
        results = []
        for event in data["events"]:
            ev_start = datetime.fromisoformat(event["start"])
            ev_end = datetime.fromisoformat(event["end"])
            # Normalize to naive if needed
            if ev_start.tzinfo is not None:
                ev_start = ev_start.replace(tzinfo=None)
            if ev_end.tzinfo is not None:
                ev_end = ev_end.replace(tzinfo=None)
            if ev_start < end and ev_end > start:
                results.append(event)
        results.sort(key=lambda e: e["start"])
        return results[:max_results]

    def create_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        data = self._load()
        event["id"] = self._new_id(data)
        event["created"] = datetime.utcnow().isoformat()
        data["events"].append(event)
        self._save(data)
        return event

    def update_event(self, event_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self._load()
        for event in data["events"]:
            if event["id"] == event_id:
                event.update(updates)
                event["updated"] = datetime.utcnow().isoformat()
                self._save(data)
                return event
        return None

    def delete_event(self, event_id: str) -> bool:
        data = self._load()
        original_len = len(data["events"])
        data["events"] = [e for e in data["events"] if e["id"] != event_id]
        if len(data["events"]) < original_len:
            self._save(data)
            return True
        return False

    def get_busy_slots(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, str]]:
        events = self.list_events(start, end)
        return [{"start": e["start"], "end": e["end"]} for e in events]


# =========================================================
# BACKEND FACTORY
# =========================================================

def _get_backend() -> LocalCalendarBackend:
    """
    Return the configured calendar backend.

    Currently returns LocalCalendarBackend always.
    To add Google Calendar or CalDAV support, implement
    the same interface (list_events, create_event, update_event,
    delete_event, get_busy_slots) and switch here based on
    CALENDAR_BACKEND env var.
    """
    backend = _cfg("CALENDAR_BACKEND", "local").lower()

    if backend == "local":
        return LocalCalendarBackend(
            path=_cfg("LOCAL_CALENDAR_PATH", "calendar.json")
        )

    # Placeholder for Google Calendar backend
    # if backend == "google":
    #     from .google_calendar_backend import GoogleCalendarBackend
    #     return GoogleCalendarBackend(
    #         credentials_path=_cfg("GOOGLE_CREDENTIALS_PATH"),
    #         calendar_id=_cfg("GOOGLE_CALENDAR_ID", "primary"),
    #     )

    # Placeholder for CalDAV backend
    # if backend == "caldav":
    #     from .caldav_backend import CalDAVBackend
    #     return CalDAVBackend(
    #         url=_cfg("CALDAV_URL"),
    #         user=_cfg("CALDAV_USER"),
    #         password=_cfg("CALDAV_PASSWORD"),
    #     )

    raise ValueError(
        f"Unsupported CALENDAR_BACKEND: '{backend}'. "
        "Use 'local', 'google', or 'caldav'."
    )


# =========================================================
# HELPERS
# =========================================================

def _parse_dt(dt_str: str) -> datetime:
    """Parse ISO 8601 date or datetime string to naive datetime."""
    dt_str = dt_str.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Could not parse datetime: '{dt_str}'. "
        "Use ISO 8601 format (e.g. '2024-11-04T14:00:00' or '2024-11-04')."
    )


def _fmt_event(event: Dict[str, Any]) -> str:
    attendees = ", ".join(event.get("attendees", [])) or "None"
    return (
        f"  ID:          {event.get('id', 'N/A')}\n"
        f"  Title:       {event.get('title', '(no title)')}\n"
        f"  Start:       {event.get('start', '')}\n"
        f"  End:         {event.get('end', '')}\n"
        f"  Location:    {event.get('location', 'None')}\n"
        f"  Attendees:   {attendees}\n"
        f"  Description: {str(event.get('description', ''))[:120]}"
    )


# =========================================================
# IMPLEMENTATIONS
# =========================================================

def list_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    calendar_id: Optional[str] = None,
    max_results: int = 50,
) -> str:
    """List calendar events within a date range."""

    now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start = _parse_dt(start_date) if start_date else now
    end = _parse_dt(end_date) if end_date else start + timedelta(days=7)

    try:
        backend = _get_backend()
        events = backend.list_events(start, end, max_results)
    except Exception as e:
        return f"ERROR listing events: {e}"

    if not events:
        return (
            f"No events found between "
            f"{start.strftime('%Y-%m-%d')} and {end.strftime('%Y-%m-%d')}."
        )

    lines = [
        f"Events from {start.strftime('%Y-%m-%d')} "
        f"to {end.strftime('%Y-%m-%d')} "
        f"({len(events)} found):\n"
        + "=" * 60
    ]

    for event in events:
        lines.append(_fmt_event(event))
        lines.append("-" * 60)

    return "\n".join(lines)


def create_event(
    title: str,
    start: str,
    end: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    timezone: str = "UTC",
    calendar_id: Optional[str] = None,
    recurrence: Optional[str] = None,
) -> str:
    """Create a new calendar event."""

    try:
        _parse_dt(start)
        _parse_dt(end)
    except ValueError as e:
        return f"ERROR: {e}"

    if _parse_dt(end) <= _parse_dt(start):
        return "ERROR: End time must be after start time."

    event = {
        "title": title,
        "start": start,
        "end": end,
        "timezone": timezone,
    }
    if description:
        event["description"] = description
    if location:
        event["location"] = location
    if attendees:
        event["attendees"] = attendees
    if recurrence:
        event["recurrence"] = recurrence

    try:
        backend = _get_backend()
        created = backend.create_event(event)
    except Exception as e:
        return f"ERROR creating event: {e}"

    return (
        f"Event created successfully.\n"
        + "=" * 60 + "\n"
        + _fmt_event(created)
    )


def update_event(
    event_id: str,
    title: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    calendar_id: Optional[str] = None,
) -> str:
    """Update fields on an existing calendar event."""

    updates = {}
    if title is not None:
        updates["title"] = title
    if start is not None:
        updates["start"] = start
    if end is not None:
        updates["end"] = end
    if description is not None:
        updates["description"] = description
    if location is not None:
        updates["location"] = location
    if attendees is not None:
        updates["attendees"] = attendees

    if not updates:
        return "ERROR: No fields provided to update."

    try:
        backend = _get_backend()
        updated = backend.update_event(event_id, updates)
    except Exception as e:
        return f"ERROR updating event: {e}"

    if updated is None:
        return f"ERROR: Event '{event_id}' not found."

    return (
        f"Event updated successfully.\n"
        + "=" * 60 + "\n"
        + _fmt_event(updated)
    )


def delete_event(
    event_id: str,
    calendar_id: Optional[str] = None,
) -> str:
    """Delete a calendar event by ID."""

    try:
        backend = _get_backend()
        # Fetch event details before deleting for confirmation output
        events = backend.list_events(
            datetime(2000, 1, 1),
            datetime(2100, 1, 1),
        )
        target = next((e for e in events if e["id"] == event_id), None)

        deleted = backend.delete_event(event_id)
    except Exception as e:
        return f"ERROR deleting event: {e}"

    if not deleted:
        return f"ERROR: Event '{event_id}' not found."

    if target:
        return (
            f"Event deleted successfully.\n"
            f"  Title: {target.get('title', '(unknown)')}\n"
            f"  Was scheduled: {target.get('start', '')} → {target.get('end', '')}"
        )

    return f"Event '{event_id}' deleted successfully."


def check_availability(
    emails: List[str],
    start: str,
    end: str,
    calendar_id: Optional[str] = None,
) -> str:
    """
    Check if a time window is free or busy.
    Uses the local calendar to find conflicting events.
    For multi-user availability, this requires a backend that
    supports querying other users' calendars (e.g. Google Calendar
    free/busy API).
    """

    try:
        start_dt = _parse_dt(start)
        end_dt = _parse_dt(end)
    except ValueError as e:
        return f"ERROR: {e}"

    if end_dt <= start_dt:
        return "ERROR: End time must be after start time."

    try:
        backend = _get_backend()
        busy_slots = backend.get_busy_slots(start_dt, end_dt)
    except Exception as e:
        return f"ERROR checking availability: {e}"

    lines = [
        f"Availability check: {start} → {end}\n"
        f"Checking for: {', '.join(emails)}\n"
        + "=" * 60
    ]

    if not busy_slots:
        lines.append(
            f"✓ AVAILABLE — No conflicts found in this window."
        )
    else:
        lines.append(
            f"✗ BUSY — {len(busy_slots)} conflict(s) found:"
        )
        for slot in busy_slots:
            lines.append(f"  Busy: {slot['start']} → {slot['end']}")

        # Suggest next available 1-hour slot after the last conflict
        last_end_str = busy_slots[-1]["end"]
        try:
            last_end = _parse_dt(last_end_str)
            suggestion_start = last_end
            suggestion_end = suggestion_start + (end_dt - start_dt)
            lines.append(
                f"\nSuggested alternative: "
                f"{suggestion_start.isoformat()} → "
                f"{suggestion_end.isoformat()}"
            )
        except Exception:
            pass

    return "\n".join(lines)


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

CALENDAR_TOOLS = {
    "list_events": {
        "func": list_events,
        "schema": ListEventsSchema,
        "description": (
            "List calendar events within a date range. "
            "Defaults to the next 7 days if no range is given."
        ),
    },
    "create_event": {
        "func": create_event,
        "schema": CreateEventSchema,
        "description": (
            "Create a new calendar event with title, start/end times, "
            "optional attendees, location, and recurrence."
        ),
    },
    "update_event": {
        "func": update_event,
        "schema": UpdateEventSchema,
        "description": (
            "Update one or more fields on an existing calendar event by ID."
        ),
    },
    "delete_event": {
        "func": delete_event,
        "schema": DeleteEventSchema,
        "description": (
            "Delete a calendar event by its ID. "
            "Returns confirmation with the deleted event's details."
        ),
    },
    "check_availability": {
        "func": check_availability,
        "schema": CheckAvailabilitySchema,
        "description": (
            "Check whether a time window is free or busy. "
            "Returns conflict details and a suggested alternative if busy."
        ),
    },
}

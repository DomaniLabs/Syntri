"""
The capture contract: what any application posts to POST /v1/capture.

The shape is deliberately thin — an instance, a kind, and an opaque
payload — so these tests are about the two things the shape has to
guarantee and cannot check itself: that the link between an interaction's
three events is expressible, and that a failure is expressible without a
missing `event_id` becoming a validation error at the wrong end.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from syntri_contracts.experience import (
    CaptureEvent,
    CaptureEventType,
    CaptureResponse,
    Observation,
)


def test_an_input_event_needs_only_an_instance_and_a_payload():
    event = CaptureEvent(
        instance_id="inst_1",
        event_type=CaptureEventType.INPUT,
        payload={"anything": "at all"},
    )

    assert event.session_id is None
    assert event.correlation_id is None
    assert event.timestamp is None, "the server fills this in"


def test_the_payload_is_opaque():
    """
    No required keys. A contract that demanded a schema here would make
    every new application a change to the contract.
    """
    event = CaptureEvent(
        instance_id="inst_1",
        event_type="outcome",
        correlation_id="obs_1",
        payload={"settled": True, "nested": {"fee": {"currency": "NGN"}}},
    )

    assert event.payload["nested"]["fee"]["currency"] == "NGN"
    assert event.event_type is CaptureEventType.OUTCOME


def test_an_event_type_outside_the_three_is_refused():
    with pytest.raises(ValidationError):
        CaptureEvent(instance_id="i", event_type="guess", payload={})


def test_a_timestamp_survives_the_round_trip():
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)

    event = CaptureEvent(
        instance_id="i", event_type="input", payload={}, timestamp=when
    )

    assert event.timestamp == when


def test_a_failure_response_carries_no_event_id():
    """
    `event_id` is nullable because every failure path returns one of
    these at HTTP 200 with nothing written.
    """
    response = CaptureResponse(
        status="unprocessable", correlation_id="obs_missing", detail="no such input"
    )

    assert response.event_id is None


def test_an_input_response_returns_its_own_id_as_the_correlation_handle():
    response = CaptureResponse(
        event_id="obs_1", status="captured", correlation_id="obs_1"
    )

    assert response.correlation_id == response.event_id


def test_an_observation_can_point_at_the_one_it_answers():
    """
    The field that closes the loop: the reply and the outcome carry the
    id of the user event that triggered them, so one interaction is one
    query rather than a session-wide guess.
    """
    inbound = Observation(instance_id="i", session_id="s", text="send 5k")

    reply = Observation(
        instance_id="i",
        session_id="s",
        text="how much?",
        correlation_id=inbound.observation_id,
    )

    assert inbound.correlation_id is None
    assert reply.correlation_id == inbound.observation_id


def test_the_correlation_id_is_written_to_the_record():
    inbound = Observation(instance_id="i", session_id="s", text="hi")
    reply = Observation(
        instance_id="i", session_id="s", text="hello",
        correlation_id=inbound.observation_id,
    )

    assert reply.model_dump()["correlation_id"] == inbound.observation_id

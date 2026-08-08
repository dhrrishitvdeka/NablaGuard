"""Context-local event and issue collection."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from .config import NablaConfig
from .events import TensorEvent
from .issues import NablaIssue

_CURRENT_SESSION: ContextVar[Session | None] = ContextVar("nablaguard_session", default=None)


@dataclass(slots=True)
class Session:
    """Shared bounded store for one instrumentation interval."""

    config: NablaConfig = field(default_factory=NablaConfig)
    events: list[TensorEvent] = field(default_factory=list)
    issues: list[NablaIssue] = field(default_factory=list)
    dropped_events: int = 0
    _token: Token[Session | None] | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> Session:
        self._token = _CURRENT_SESSION.set(self)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._token is not None:
            _CURRENT_SESSION.reset(self._token)
            self._token = None

    def emit_event(self, event: TensorEvent) -> None:
        """Add metadata while enforcing the configured memory bound."""

        if len(self.events) >= self.config.max_events:
            self.dropped_events += 1
            return
        self.events.append(event)

    def emit_issue(self, issue: NablaIssue) -> None:
        """Add an issue to this session."""

        self.issues.append(issue)


def current_session() -> Session | None:
    """Return the active context-local session, if any."""

    return _CURRENT_SESSION.get()


def emit_issue(issue: NablaIssue) -> None:
    """Emit into the active session when called by a standalone subsystem."""

    session = current_session()
    if session is not None:
        session.emit_issue(issue)

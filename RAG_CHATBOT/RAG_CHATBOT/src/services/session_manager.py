from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque
import uuid

@dataclass
class ChatSession:
    """Individual chat session with its own memory"""
    
    session_id: str
    window_length: int = 3
    messages: Deque[Dict[str, str]] = field(default_factory=deque)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.messages = deque(maxlen=self.window_length * 2)  # user + assistant pairs

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session memory"""
        self.messages.append({"role": role, "content": content})
        self.last_activity = datetime.now()

    def get_context(self) -> List[Dict[str, str]]:
        """Get conversation context for this session"""
        return list(self.messages)

    def clear(self) -> None:
        """Clear the session memory"""
        self.messages.clear()

    def get_summary(self) -> Dict[str, any]:
        """Get session summary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": len(self.messages)
        }

class SessionManager:
    """Manages multiple chat sessions"""
    
    def __init__(self, default_window_length: int = 3):
        self.sessions: Dict[str, ChatSession] = {}
        self.default_window_length = default_window_length

    def create_session(self, session_id: str = None) -> ChatSession:
        """Create a new chat session"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id in self.sessions:
            return self.sessions[session_id]
            
        session = ChatSession(
            session_id=session_id,
            window_length=self.default_window_length
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a specific session"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, any]]:
        """List all active sessions"""
        return [
            session.get_summary()
            for session in self.sessions.values()
        ]

    def cleanup_inactive_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than max_age_hours"""
        now = datetime.now()
        deleted = 0
        
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            session = self.sessions[session_id]
            age_hours = (now - session.last_activity).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                self.delete_session(session_id)
                deleted += 1
                
        return deleted

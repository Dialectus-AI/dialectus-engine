CREATE TABLE IF NOT EXISTS debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    format TEXT NOT NULL,
    participants TEXT NOT NULL,  -- JSON string
    final_phase TEXT NOT NULL,
    total_rounds INTEGER NOT NULL,
    saved_at TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    total_debate_time_ms INTEGER NOT NULL,  -- Total debate duration in milliseconds
    scores TEXT,  -- JSON string
    context_metadata TEXT,  -- JSON string
    user_id INTEGER,  -- Foreign key to users.id, NULL for unauthenticated debates
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);
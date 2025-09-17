CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id INTEGER NOT NULL,
    speaker_id TEXT NOT NULL,
    position TEXT NOT NULL,
    phase TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    metadata TEXT,  -- JSON string
    FOREIGN KEY (debate_id) REFERENCES debates (id) ON DELETE CASCADE
);
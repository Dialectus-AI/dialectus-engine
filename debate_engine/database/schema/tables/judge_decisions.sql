CREATE TABLE IF NOT EXISTS judge_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id INTEGER NOT NULL,
    judge_model TEXT NOT NULL,
    judge_provider TEXT NOT NULL,
    winner_id TEXT NOT NULL,
    winner_margin REAL NOT NULL,
    overall_feedback TEXT,
    reasoning TEXT,
    generation_time_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
);
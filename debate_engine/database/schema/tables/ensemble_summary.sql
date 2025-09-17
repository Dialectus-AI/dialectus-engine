CREATE TABLE IF NOT EXISTS ensemble_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id INTEGER NOT NULL UNIQUE,
    final_winner_id TEXT NOT NULL,
    final_margin REAL NOT NULL,
    ensemble_method TEXT NOT NULL DEFAULT 'majority',
    num_judges INTEGER NOT NULL,
    consensus_level REAL,
    summary_reasoning TEXT,
    summary_feedback TEXT,
    participating_judge_decision_ids TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
);
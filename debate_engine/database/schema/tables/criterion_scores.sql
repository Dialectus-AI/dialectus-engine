CREATE TABLE IF NOT EXISTS criterion_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    judge_decision_id INTEGER NOT NULL,
    criterion TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    score REAL NOT NULL,
    feedback TEXT,
    FOREIGN KEY (judge_decision_id) REFERENCES judge_decisions (id) ON DELETE CASCADE
);
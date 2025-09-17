CREATE TABLE IF NOT EXISTS tournament_judges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tournament_id INTEGER NOT NULL,
    judge_model_id TEXT NOT NULL,
    judge_provider TEXT NOT NULL,
    FOREIGN KEY (tournament_id) REFERENCES tournaments (id) ON DELETE CASCADE
);
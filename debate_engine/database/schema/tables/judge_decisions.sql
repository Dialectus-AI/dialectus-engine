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
    cost REAL,  -- Cost in USD for OpenRouter judge models (NULL for Ollama/free models)
    generation_id TEXT,  -- OpenRouter generation ID for cost queries
    cost_queried_at DATETIME,  -- When cost was successfully retrieved
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
);
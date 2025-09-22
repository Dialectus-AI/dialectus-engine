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
    cost REAL,  -- Cost in USD for OpenRouter models (NULL for Ollama/free models)
    generation_id TEXT,  -- OpenRouter generation ID for cost queries
    cost_queried_at DATETIME,  -- When cost was successfully retrieved
    FOREIGN KEY (debate_id) REFERENCES debates (id) ON DELETE CASCADE
);
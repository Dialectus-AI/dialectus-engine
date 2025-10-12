"""Topic generation themes and tones for debate topic generation."""

from enum import Enum


class TopicTheme(Enum):
    """High-level themes for debate topic generation."""

    HUMOR = "humor"
    PHILOSOPHICAL = "philosophical"
    TECHNOLOGY = "technology"
    CURRENT_AFFAIRS = "current_affairs"
    SCIENCE = "science"
    HISTORICAL = "historical"
    POP_CULTURE = "pop_culture"
    SOCIAL_CULTURAL = "social_cultural"
    PROFESSIONAL_BUSINESS = "professional_business"
    SPECULATIVE = "speculative"


class TopicTone(Enum):
    """Specific tones/sub-themes for more granular topic generation."""

    # Humor sub-tones
    ABSURD = "absurd"
    PLAYFUL = "playful"
    SATIRICAL = "satirical"

    # Philosophical sub-tones
    EXISTENTIAL = "existential"
    ETHICAL = "ethical"
    METAPHYSICAL = "metaphysical"
    EPISTEMOLOGICAL = "epistemological"

    # Technology sub-tones
    AI_ML = "ai_ml"
    PRIVACY_SECURITY = "privacy_security"
    INNOVATION = "innovation"
    DIGITAL_LIFE = "digital_life"

    # Current Affairs sub-tones
    POLICY = "policy"
    GOVERNANCE = "governance"
    SOCIAL_ISSUES = "social_issues"
    INTERNATIONAL = "international"

    # Science sub-tones
    RESEARCH_ETHICS = "research_ethics"
    METHODOLOGY = "methodology"
    THEORIES = "theories"
    EXPLORATION = "exploration"

    # Historical sub-tones
    INTERPRETATIONS = "interpretations"
    COUNTERFACTUALS = "counterfactuals"
    LEGACY = "legacy"
    CAUSATION = "causation"

    # Pop Culture sub-tones
    ENTERTAINMENT = "entertainment"
    MEDIA = "media"
    FANDOMS = "fandoms"
    TRENDS = "trends"

    # Social & Cultural sub-tones
    NORMS = "norms"
    RELATIONSHIPS = "relationships"
    ETIQUETTE = "etiquette"
    TRADITIONS = "traditions"

    # Professional/Business sub-tones
    WORKPLACE = "workplace"
    ECONOMICS = "economics"
    CAREER = "career"
    MANAGEMENT = "management"

    # Speculative sub-tones
    FUTURE_PREDICTIONS = "future_predictions"
    WHAT_IFS = "what_ifs"
    EMERGING_TECH = "emerging_tech"
    SOCIETAL_CHANGE = "societal_change"


# Mapping of themes to their applicable tones
THEME_TONES: dict[TopicTheme, list[TopicTone]] = {
    TopicTheme.HUMOR: [
        TopicTone.ABSURD,
        TopicTone.PLAYFUL,
        TopicTone.SATIRICAL,
    ],
    TopicTheme.PHILOSOPHICAL: [
        TopicTone.EXISTENTIAL,
        TopicTone.ETHICAL,
        TopicTone.METAPHYSICAL,
        TopicTone.EPISTEMOLOGICAL,
    ],
    TopicTheme.TECHNOLOGY: [
        TopicTone.AI_ML,
        TopicTone.PRIVACY_SECURITY,
        TopicTone.INNOVATION,
        TopicTone.DIGITAL_LIFE,
    ],
    TopicTheme.CURRENT_AFFAIRS: [
        TopicTone.POLICY,
        TopicTone.GOVERNANCE,
        TopicTone.SOCIAL_ISSUES,
        TopicTone.INTERNATIONAL,
    ],
    TopicTheme.SCIENCE: [
        TopicTone.RESEARCH_ETHICS,
        TopicTone.METHODOLOGY,
        TopicTone.THEORIES,
        TopicTone.EXPLORATION,
    ],
    TopicTheme.HISTORICAL: [
        TopicTone.INTERPRETATIONS,
        TopicTone.COUNTERFACTUALS,
        TopicTone.LEGACY,
        TopicTone.CAUSATION,
    ],
    TopicTheme.POP_CULTURE: [
        TopicTone.ENTERTAINMENT,
        TopicTone.MEDIA,
        TopicTone.FANDOMS,
        TopicTone.TRENDS,
    ],
    TopicTheme.SOCIAL_CULTURAL: [
        TopicTone.NORMS,
        TopicTone.RELATIONSHIPS,
        TopicTone.ETIQUETTE,
        TopicTone.TRADITIONS,
    ],
    TopicTheme.PROFESSIONAL_BUSINESS: [
        TopicTone.WORKPLACE,
        TopicTone.ECONOMICS,
        TopicTone.CAREER,
        TopicTone.MANAGEMENT,
    ],
    TopicTheme.SPECULATIVE: [
        TopicTone.FUTURE_PREDICTIONS,
        TopicTone.WHAT_IFS,
        TopicTone.EMERGING_TECH,
        TopicTone.SOCIETAL_CHANGE,
    ],
}


# Theme descriptions for prompt generation (no examples!)
THEME_DESCRIPTIONS: dict[TopicTheme, str] = {
    TopicTheme.HUMOR: "lighthearted and amusing, designed to be entertaining while still being debatable. The topic should invite clever arguments and witty reasoning.",
    TopicTheme.PHILOSOPHICAL: "exploring deep questions about existence, knowledge, morality, or reality. The topic should engage with fundamental concepts and abstract reasoning.",
    TopicTheme.TECHNOLOGY: "addressing modern technological developments, their impact on society, or emerging innovations. The topic should be relevant to contemporary tech discourse.",
    TopicTheme.CURRENT_AFFAIRS: "dealing with present-day real-world issues, policies, or events. The topic should be timely and relevant to current societal discussions.",
    TopicTheme.SCIENCE: "focused on scientific research, methods, discoveries, or their implications. The topic should engage with empirical inquiry or scientific reasoning.",
    TopicTheme.HISTORICAL: "examining past events, figures, or movements and their interpretations. The topic should invite analysis of historical significance or causation.",
    TopicTheme.POP_CULTURE: "centered on entertainment, media, or cultural phenomena in the public consciousness. The topic should be accessible and culturally relevant.",
    TopicTheme.SOCIAL_CULTURAL: "exploring social norms, cultural practices, or human interactions. The topic should examine how people relate to each other and their communities.",
    TopicTheme.PROFESSIONAL_BUSINESS: "addressing workplace practices, economic principles, or professional development. The topic should be relevant to career and business contexts.",
    TopicTheme.SPECULATIVE: "imagining future possibilities, hypothetical scenarios, or predictions about what may come. The topic should encourage forward-thinking analysis.",
}


# Tone descriptions for more specific prompt refinement (no examples!)
TONE_DESCRIPTIONS: dict[TopicTone, str] = {
    # Humor tones
    TopicTone.ABSURD: "ridiculous or nonsensical in a way that highlights the humor in everyday contradictions or illogical situations",
    TopicTone.PLAYFUL: "whimsical and fun, inviting creative and spirited argumentation without being too serious",
    TopicTone.SATIRICAL: "using irony or mockery to critique societal norms, behaviors, or institutions in a humorous way",

    # Philosophical tones
    TopicTone.EXISTENTIAL: "questioning the nature of existence, meaning, purpose, or human condition",
    TopicTone.ETHICAL: "examining moral principles, right and wrong, or the foundations of ethical behavior",
    TopicTone.METAPHYSICAL: "exploring the fundamental nature of reality, being, or the universe",
    TopicTone.EPISTEMOLOGICAL: "investigating the nature of knowledge, belief, truth, or how we know what we know",

    # Technology tones
    TopicTone.AI_ML: "centered on artificial intelligence, machine learning, or automated decision-making systems",
    TopicTone.PRIVACY_SECURITY: "addressing data protection, surveillance, cybersecurity, or digital privacy concerns",
    TopicTone.INNOVATION: "focused on new technological developments, breakthroughs, or disruptive technologies",
    TopicTone.DIGITAL_LIFE: "examining how technology shapes daily life, communication, or social interaction",

    # Current Affairs tones
    TopicTone.POLICY: "addressing governmental policies, regulations, or legislative approaches to contemporary problems",
    TopicTone.GOVERNANCE: "examining systems of government, democratic processes, or institutional structures",
    TopicTone.SOCIAL_ISSUES: "dealing with societal challenges, inequality, justice, or community concerns",
    TopicTone.INTERNATIONAL: "focused on global affairs, international relations, or cross-border issues",

    # Science tones
    TopicTone.RESEARCH_ETHICS: "examining moral considerations in scientific research, experimentation, or methodology",
    TopicTone.METHODOLOGY: "focused on scientific methods, experimental design, or approaches to inquiry",
    TopicTone.THEORIES: "addressing competing scientific theories, paradigms, or explanatory frameworks",
    TopicTone.EXPLORATION: "centered on scientific discovery, space exploration, or pushing the boundaries of knowledge",

    # Historical tones
    TopicTone.INTERPRETATIONS: "examining different ways of understanding or analyzing historical events or figures",
    TopicTone.COUNTERFACTUALS: "exploring alternative historical scenarios or what might have happened differently",
    TopicTone.LEGACY: "assessing the long-term impact or significance of historical events, movements, or individuals",
    TopicTone.CAUSATION: "analyzing what caused historical events or how different factors contributed to outcomes",

    # Pop Culture tones
    TopicTone.ENTERTAINMENT: "focused on movies, television, music, or other forms of popular entertainment",
    TopicTone.MEDIA: "examining news media, social media, content creation, or information dissemination",
    TopicTone.FANDOMS: "addressing fan communities, fandom culture, or audience engagement with media",
    TopicTone.TRENDS: "centered on viral phenomena, cultural trends, or shifting public tastes",

    # Social & Cultural tones
    TopicTone.NORMS: "examining unwritten social rules, expected behaviors, or cultural conventions",
    TopicTone.RELATIONSHIPS: "focused on interpersonal dynamics, romantic relationships, or social connections",
    TopicTone.ETIQUETTE: "addressing proper conduct, manners, or appropriate behavior in various contexts",
    TopicTone.TRADITIONS: "examining cultural practices, rituals, or customs passed through generations",

    # Professional/Business tones
    TopicTone.WORKPLACE: "focused on office culture, work practices, or the modern work environment",
    TopicTone.ECONOMICS: "addressing economic principles, market forces, or financial systems",
    TopicTone.CAREER: "examining professional development, career paths, or work-life considerations",
    TopicTone.MANAGEMENT: "centered on leadership, organizational structure, or business strategy",

    # Speculative tones
    TopicTone.FUTURE_PREDICTIONS: "making informed predictions about likely future developments or outcomes",
    TopicTone.WHAT_IFS: "exploring hypothetical scenarios or imagining alternative possibilities",
    TopicTone.EMERGING_TECH: "anticipating the impact of nascent technologies or scientific breakthroughs",
    TopicTone.SOCIETAL_CHANGE: "examining potential shifts in social structures, values, or human organization",
}


def get_theme_display_name(theme: TopicTheme) -> str:
    """Get human-readable display name for a theme."""
    return theme.value.replace("_", " ").title()


def get_tone_display_name(tone: TopicTone) -> str:
    """Get human-readable display name for a tone."""
    return tone.value.replace("_", " ").title()

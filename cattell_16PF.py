# Mapping relationships between models based on psychological insights
OCEAN_to_16PF = {
    "Openness": ["Abstractedness"],
    "Conscientiousness": ["Rule-Consciousness", "Perfectionism", "Self-Reliance"],
    "Extraversion": ["Warmth", "Social Boldness", "Liveliness"],
    "Agreeableness": ["Sensitivity", "Warmth", "Empathy"],
    "Neuroticism": ["Apprehension", "Emotional Stability", "Tension"]
}

MBTI_to_16PF = {
    "E": ["Warmth", "Social Boldness", "Liveliness"],
    "I": ["Self-Reliance", "Perfectionism"],
    "S": ["Rule-Consciousness"],
    "N": ["Abstractedness"],
    "T": ["Reasoning", "Dominance"],
    "F": ["Sensitivity", "Empathy"],
    "J": ["Perfectionism", "Rule-Consciousness"],
    "P": ["Openness to Change"]
}

# Initialize base 16PF scores dictionary
Cattell_16PF_scores = {
    "Warmth": 0, "Reasoning": 0, "Emotional Stability": 0, "Dominance": 0,
    "Liveliness": 0, "Rule-Consciousness": 0, "Social Boldness": 0,
    "Sensitivity": 0, "Vigilance": 0, "Abstractedness": 0, "Privateness": 0,
    "Apprehension": 0, "Openness to Change": 0, "Self-Reliance": 0,
    "Perfectionism": 0, "Tension": 0
}

def predict_cattell_16PF(OCEAN_scores, MBTI_type):
    # Assign scores based on OCEAN influence
    for trait, score in OCEAN_scores.items():
        related_factors = OCEAN_to_16PF.get(trait, [])
        for factor in related_factors:
            if factor in Cattell_16PF_scores:
                Cattell_16PF_scores[factor] += score  # Simple addition for trait influence

    # Assign scores based on MBTI influence
    for dimension in MBTI_type:
        related_factors = MBTI_to_16PF.get(dimension, [])
        for factor in related_factors:
            if factor in Cattell_16PF_scores:
                Cattell_16PF_scores[factor] += 0.5  # Heuristic weight for MBTI influence

    # Normalize scores to 0-1 for consistency
    max_score = max(Cattell_16PF_scores.values())
    if max_score > 0:  # Avoid division by zero
        for factor in Cattell_16PF_scores:
            Cattell_16PF_scores[factor] /= max_score  # Scale scores

    return Cattell_16PF_scores

# Sample OCEAN and MBTI inputs
OCEAN_sample = {
    "Openness": 0.7,
    "Conscientiousness": 0.8,
    "Extraversion": 0.6,
    "Agreeableness": 0.5,
    "Neuroticism": 0.3
}

MBTI_type = "ENTP"  # Example MBTI input

# Run prediction
predicted_16PF = predict_cattell_16PF(OCEAN_sample, MBTI_type)
print("Predicted Cattell's 16PF Scores:")
for factor, score in predicted_16PF.items():
    print(f"{factor}: {score:.2f}")

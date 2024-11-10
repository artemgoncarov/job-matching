def ocean_to_mbti(ocean_scores):
    """
    Переводит оценки по системе OCEAN в MBTI тип.
    
    Параметры:
    ocean_scores (dict): словарь с ключами 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'
    
    Возвращает:
    str: тип MBTI
    """
    openness = ocean_scores['Openness']
    conscientiousness = ocean_scores['Conscientiousness']
    extraversion = ocean_scores['Extraversion']
    agreeableness = ocean_scores['Agreeableness']
    neuroticism = ocean_scores['Neuroticism']
    
    # Определяем E/I
    ei = 'E' if extraversion >= 0.5 else 'I'
    
    # Определяем S/N на основе Openness и Conscientiousness
    if openness >= 0.5 and conscientiousness < 0.5:
        sn = 'N'
    elif openness < 0.5 and conscientiousness >= 0.5:
        sn = 'S'
    elif openness >= 0.5 and conscientiousness >= 0.5:
        sn = 'N'  # Открытость превалирует, если обе высокие
    else:
        sn = 'S'  # Ориентация на конкретные детали
    
    # Определяем T/F на основе Agreeableness
    tf = 'F' if agreeableness >= 0.5 else 'T'
    
    # Определяем J/P на основе Conscientiousness и Neuroticism
    if conscientiousness >= 0.5 and neuroticism < 0.5:
        jp = 'J'
    elif conscientiousness < 0.5 and neuroticism >= 0.5:
        jp = 'P'
    else:
        jp = 'J' if conscientiousness >= 0.5 else 'P'
    
    # Собираем MBTI тип
    mbti_type = ei + sn + tf + jp
    return mbti_type
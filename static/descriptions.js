const description = {
    "INTJ": {
        "Основные черты": "Независимость, стремление к целям, стратегическое мышление.",
        "Особенности": "Предпочитают работать над долгосрочными проектами, склонны самостоятельно принимать решения и планировать стратегию, что делает их отличными лидерами в проектных командах. Идеально подходят для разработки новых концепций, любят ясность задач и четкие цели."
    },
    "INTP": {
        "Основные черты": "Аналитичность, любознательность, склонность к теоретизированию.",
        "Особенности": "Обычно предпочитают работать над задачами, требующими глубокого анализа и исследования. Умеют находить инновационные решения и строить теории, однако могут работать эффективнее в условиях, где есть свобода действий и нет строгих рамок. В команде склонны брать роль аналитика или исследователя."
    },
    "ENTJ": {
        "Основные черты": "Лидерство, решительность, логическое мышление.",
        "Особенности": "Хорошо справляются с руководящими позициями, умеют распределять задачи и вдохновлять команду на достижения. Стремятся к эффективности и достижению максимального результата. В команде часто выполняют роль координатора, организуя работу других на высоком уровне и обеспечивая слаженность."
    },
    "ENTP": {
        "Основные черты": "Предприимчивость, креативность, способность к импровизации.",
        "Особенности": "Любят генерировать новые идеи и легко адаптируются к изменениям. В команде они привносят оригинальные решения и могут стимулировать обсуждения. Их способность видеть ситуацию с разных сторон полезна в мозговых штурмах, но им лучше работать там, где есть свобода для экспериментов."
    },
    "INFJ": {
        "Основные черты": "Эмпатия, забота о других, стремление к гармонии.",
        "Особенности": "Идеально подходят для задач, где важно учитывать интересы людей. Их стремление к гармонии помогает в разрешении конфликтов и налаживании отношений в команде. Они могут эффективно работать в роли наставников, помогая другим раскрыть свои сильные стороны."
    },
    "INFP": {
        "Основные черты": "Мечтательность, искренность, чувствительность.",
        "Особенности": "Ориентированы на работу, которая связана с личными ценностями. Им важно работать над проектами, которые соответствуют их идеалам. В команде создают комфортную и поддерживающую атмосферу, могут выступать в роли генераторов идей или мотиваторов, поддерживающих моральный дух."
    },
    "ENFJ": {
        "Основные черты": "Харизма, забота о других, социальные навыки.",
        "Особенности": "Часто берут на себя роль лидера или вдохновителя в команде. Их способность мотивировать других делает их идеальными менеджерами и наставниками. Сфокусированы на взаимодействии и стремятся создать команду, где каждый чувствует себя ценным и поддержанным."
    },
    "ENFP": {
        "Основные черты": "Оптимизм, энтузиазм, творческое мышление.",
        "Особенности": "Вдохновляют и мотивируют команду, часто предлагают инновационные подходы к задачам. В команде любят экспериментировать и пробовать что-то новое, привносят энергию и создают приятную атмосферу, хотя иногда могут легко отвлекаться, если не видят ясной цели."
    },
    "ISTJ": {
        "Основные черты": "Ответственность, практичность, внимательность.",
        "Особенности": "Привыкли придерживаться четких инструкций и следовать плану. В работе ориентированы на стабильность и аккуратность. В команде являются надежной опорой, склонны придерживаться расписания и следить за деталями, чтобы избежать ошибок."
    },
    "ISFJ": {
        "Основные черты": "Доброта, терпеливость, ориентированность на других.",
        "Особенности": "Имеют склонность к поддержанию гармонии в коллективе, что делает их хорошими командными игроками. В работе отличаются вниманием к деталям и ответственностью. Помогают коллегам, поддерживают комфортную атмосферу и всегда готовы предложить помощь."
    },
    "ESTJ": {
        "Основные черты": "Организаторские способности, практичность, уверенность.",
        "Особенности": "Отлично справляются с руководящими ролями, где требуется структура и порядок. В работе следуют установленным правилам, стремятся к эффективности и результатам. Их уверенность и организаторские способности позволяют им контролировать и координировать команду, обеспечивая выполнение всех задач."
    },
    "ESFJ": {
        "Основные черты": "Заботливость, общительность, стремление помочь.",
        "Особенности": "Создают комфортные условия для команды, поддерживают каждого, кто нуждается в помощи. Часто выступают связующим звеном между членами команды, улучшая коммуникацию. Их стремление к гармонии способствует налаживанию позитивных отношений внутри коллектива."
    },
    "ISTP": {
        "Основные черты": "Находчивость, склонность к действиям, практическое мышление.",
        "Особенности": "Любят решать проблемы на месте и работать с конкретными задачами. В команде предпочитают работать автономно и часто выполняют роль тех, кто быстро справляется с задачами, требующими конкретных решений и действий."
    },
    "ISFP": {
        "Основные черты": "Творческий подход, чувствительность, спонтанность.",
        "Особенности": "Предпочитают работать в спокойной, неформальной атмосфере, что позволяет раскрыть их творческий потенциал. В команде добавляют уникальные идеи, хотя предпочитают выполнять задачи по своему собственному графику."
    },
    "ESTP": {
        "Основные черты": "Авантюризм, решительность, практическое мышление.",
        "Особенности": "Умеют справляться с кризисами и быстро принимать решения. В команде они могут быть отличными инициаторами действий, способными предложить нестандартные решения в трудных ситуациях, склонны к практическому подходу."
    },
    "ESFP": {
        "Основные черты": "Общительность, жизнерадостность, спонтанность.",
        "Особенности": "Любят работать с людьми и поднимать настроение в команде. Создают живую атмосферу и мотивируют других своими идеями. Их гибкость и энергия помогают легко адаптироваться к изменениям и привносят позитивный настрой в коллектив."
    }
}
import spacy
import textacy.extract

# Загрузка английской NLP-модели
nlp = spacy.load('en_core_web_lg')

# Текст для анализа
text = """London is the capital and most populous city of England and 
the United Kingdom. Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
"""

# Парсинг текста с помощью spaCy. Эта команда запускает целый конвейер  
doc = nlp(text)

# в переменной 'doc' теперь содержится обработанная версия текста
# мы можем делать с ней все что угодно!
# например, распечатать все обнаруженные именованные сущности
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")

# Если токен является именем, заменяем его словом "REDACTED" 
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == "PERSON":
        return "[REDACTED] "
    else:
        return token.text

# Проверка всех сущностей
def scrub(text):
    doc = nlp(text)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
    tokens = map(replace_name_with_placeholder, doc)
    return "".join(tokens)

s = """
In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence". In 1957, Noam Chomsky’s 
Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.
"""

print(scrub(s))

text = """London is the capital and most populous city of England and the United Kingdom.  
Standing on the River Thames in the south east of the island of Great Britain, 
London has been a major settlement for two millennia.  It was founded by the Romans, 
who named it Londinium.
"""

# Анализ
doc = nlp(text)

# Извлечение полуструктурированных выражений со словом London
statements = textacy.extract.semistructured_statements(doc, entity="London", cue="be")

# Вывод результатов
print("Here are the things I know about London:")

for statement in statements:
    subject, verb, fact = statement
    print(f" - {fact}")
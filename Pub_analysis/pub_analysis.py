import pickle

KEYWORD = 'English'
KEYWORD = 'French'
KEYWORD = 'German'
KEYWORD = 'Dutch'
KEYWORD = 'Italian'

KEYWORD = 'Mandarin'
KEYWORD = 'Cantonese'
KEYWORD = 'Korean'
KEYWORD = 'Japanese'

KEYWORDS = [
            'English',
            'French',
            'German',
            'Dutch',
            'Italian',
            'Swedish',
            'Spanish',
            'Swiss',
            'Chinese',
            'Mandarin',
            'Cantonese',
            'Minnan',
            'Korean',
            'Japanese',
            'Vietnamese',
            ]

def parse_one(lang):
    lang2 = '{' + lang[0] + '}' + lang[1:]
    # print(lang2)

    # fpin = open('anthology.bib', 'r', encoding='utf-8')
    fpin = open('anthology+abstracts.bib', 'r', encoding='utf-8')

    # fpout = open('refined_' + lang + '.txt', 'w', encoding='utf-8')
    fpout = open('titleabs_' + lang + '.txt', 'w', encoding='utf-8')

    dict_title = {}

    START_TOKEN = '@'
    END_TOKEN = '}'

    b_start = False
    valuestr = ''
    for i, line in enumerate(fpin):
        # if i % 10000 == 0:
        #     print(i)
        valuestr = valuestr + line
        if START_TOKEN in line and b_start == False:
            b_start = True
            keystr = line.strip()[0:-1].split('{')[1]
        elif len(line.strip()) == 1 and line.strip() == END_TOKEN and b_start:

            if lang.lower() in valuestr.lower() or lang2.lower() in valuestr.lower():
                dict_title[keystr] = valuestr
                fpout.writelines(valuestr)

            valuestr = ''
            b_start = False

    print(len(dict_title.keys()))
    print()

def parse_all():
    for lang in KEYWORDS:
        print(lang)
        parse_one(lang)

# parse_one('low-resource')
parse_all()
content = open('requirements.txt', encoding='utf-16').read()
open('requirements.txt', 'w', encoding='utf-8').write(content)
print("done")
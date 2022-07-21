inString = input('give letters')
outString = ''
for i in inString:
    if not(('a' in i) or ('t' in i) or ('c' in i) or ('g' in i)):
         print(('a' in i))
         print(f'oopsy woopsy: {i}')
         exit()
    else:
        if i=='a':
            outString += '00'
        elif i=='t':
            outString += '01'
        elif i=='c':
            outString += '10'
        elif i=='g':
            outString += '11'

print(outString)



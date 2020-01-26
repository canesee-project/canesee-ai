list1=[]
list2=[]
filepath="word_countss.txt"
with open(filepath) as fp:
    
   for line in fp:
       
      list1.append(line.replace("b",''))
      

for line in list1:
       list2.append(line.replace("'",''))


print(list2[0])
i=0
file=open("word_counts.txt", "w")

for l in list2:
    file.write(l)



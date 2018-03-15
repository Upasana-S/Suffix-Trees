from suffix_tree import SuffixTree, GST
from document import Directory
#f = open("AesopTales.txt")
Aesop=Directory()
string=Aesop.documentify("AesopTales.txt")
tree= SuffixTree(string)

# all occurences of a substring in all the stories
ip=input()
indices=tree.all_occurences(ip)
print " all occurences of",ip,": ",indices
count=0
for i in indices:
    title=''
    j=i
    for doc in Aesop.docs:
        if(i> doc.start and i< doc.end):
            title=doc.title
            j-=doc.start
    print j, title,  string[i:i+40],"\n\n"
    count+=1
print"(",count,"occurences )"

#first occurence/closest match in each story
query=input()

for doc in Aesop.docs:
    if doc.start-doc.end >=0:
	continue
    story= string[doc.start:doc.end]
    st=SuffixTree(story)
    i=st.find(query)
    print i, doc.title,  string[i:i+10], '\n',query, "\n\n"





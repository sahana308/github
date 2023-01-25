class Solution:
    def findSubString(self, str):
        a=str
        n=len(a)
        b=set(a)
        c=len(b)
        s={}
        i=0
        j=0
        k=0
        sol=10**9
        while i<n and j<n:
            if a[i] not in s.keys():
                s[a[i]]=1
                k+=1
            else:
                s[a[i]]+=1
            i+=1
            while s[a[j]]>1:
                s[a[j]]-=1
                j+=1
            if k==c:
                sol=min(sol,i-j)
                
        return sol
        

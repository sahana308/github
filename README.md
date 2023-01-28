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
  
  
  LEETCODE (Date: 25-01-2023)
 1) UNIVALUED BINARY TREE
  class Solution:
    def isUnivalTree(self,t):
        if not t: return True
        if t.left:
            if t.left.val != t.val :
                return False 
 2) Invert Tree               
   class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
            if not root:
                return None
            left = root.left
            right = root.right
            root.left = self.invertTree(right)
            root.right = self.invertTree(left)
            return root

3) Balanced Binary Tree
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool :
        def dfs(root):
            if not root: 
                return [True,0]
            left = dfs(root.left)
            right = dfs(root.right)
            balance = (left[0] and right[0] and abs(left[1] - right[1])<=1)
            return[balance, 1+max(left[1],right[1])]
       return dfs(root)[0] 
    
4) Binary Tree Paths        
 class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def backtrack(node, path, t):
            ch = (node.left, node.right)
            
            path.append(str(node.val))

            if not any(ch): 
                t.append('->'.join(path))
                
            for ch in filter(None, ch):
                backtrack(ch, path, t)
                
            path.pop()
                    
            return t
        
        return backtrack(root, [], [])
        if t.right:
            if t.right.val != t.val :
                return False
        return self.isUnivalTree(t.left) and self.isUnivalTree(t.right)

LEETCODE(28-01-2023)
1)Maximum Binary Tree
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums: return
        root = max(nums)
        i = nums.index(root)
        node = TreeNode(root)
        node.left = self.constructMaximumBinaryTree(nums[0:i])
        node.right = self.constructMaximumBinaryTree(nums[i + 1:])
        return node
2)Binary Tree Inorder Traversal
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        sol =[]
        def inorder(root):
            if not root: return 
            inorder(root.left)
            sol.append(root.val)
            inorder(root.right)
        inorder(root)
        return sol
3)Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        sol=[]
        def preorder(root):
            if not root: return
            sol.append(root.val)
            preorder(root.left)
            preorder(root.right)
        preorder(root)
        return sol
4)Binary Tree Postorder Traversal
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        sol=[]
        def postorder(root):
            if not root: return 
            postorder(root.left)
            postorder(root.right)
            sol.append(root.val)
        postorder(root)
        return sol
     

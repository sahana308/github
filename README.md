 LEETCODE (Date: 27-01-2023)
1)class Solution:
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
  
 2) UNIVALUED BINARY TREE
  class Solution:
    def isUnivalTree(self,t):
        if not t: return True
        if t.left:
            if t.left.val != t.val :
                return False 
 3) Invert Tree               
   class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
            if not root:
                return None
            left = root.left
            right = root.right
            root.left = self.invertTree(right)
            root.right = self.invertTree(left)
            return root

4) Balanced Binary Tree
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
    
5) Binary Tree Paths        
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
     
LEETCODE(29-01-2023)
1)Evaluate Boolean Binary Tree
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return False
        if(root.val==2):
            return self.evaluateTree(root.left) or self.evaluateTree(root.right)
        if(root.val==3):
            return self.evaluateTree(root.left) and self.evaluateTree(root.right)
        return root.val
 2)Minimum Depth of Binary Tree
 class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0

        left = self.minDepth(root.left)
        right = self.minDepth(root.right)

        return 1 + (min(left,right) or max(left,right))
  3)Balanced Binary Tree
  class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool :
        def dfs(root):
            if not root: 
                return [True,0]
            left = dfs(root.left)
            right = dfs(root.right)
            balance = (left[0] and right[0] and abs(left[1] - right[1])<=1)
            return[balance, 1+max(left[1],right[1])]
        return dfs(root)
  4)Binary Tree Paths
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
   5)Combination Sum
   class Solution(object):
    def combinationSum(self, candidates, target):
        res = []
        self.dfs(candidates, target, [], res)
        return res
    
    def dfs(self, nums, target, path, res):
        if target < 0:
            return 
        if target == 0:
            res.append(path)
            return 
        for i in range(len(nums)):
            self.dfs(nums[i:], target-nums[i], path+[nums[i]], res)
    6)Pow(x, n)
    class Solution:
    def myPow(self, x: float, n: int) -> float:
        return x ** n
LEETCODE(30-01-2023)
1)Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums)-1
        
        while l<=r:
            mid = (l+r)//2
            if nums[mid]==target:
                return mid
            elif nums[mid]>target:
                r = mid-1
            else:
                l = mid+1
        return -1
2)Diameter of Binary Tree
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.dia=0
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            if left + right > self.dia :
                self.dia = left + right
            return max(left, right) + 1
        dfs(root)
        return self.dia
 3)Merge Two Sorted Lists
 class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = d = ListNode()
        while list1 and list2:               
            if list1.val < list2.val:
                cur.next = list1
                list1, cur = list1.next, list1
            else:
                cur.next = list2
                list2, cur = list2.next, list2
                
        if list1 or list2:
            cur.next = list1 if list1 else list2
            
        return d.next
   4) Rotate Image
   class Solution:

    def rotate(self, M: List[List[int]]) -> None:

        n = len(M)

        depth = n // 2

        for i in range(depth):

            rlen, opp = n - 2 * i - 1, n - 1 - i

            for j in range(rlen):

                temp = M[i][i+j]

                M[i][i+j] = M[opp-j][i]

                M[opp-j][i] = M[opp][opp-j]

                M[opp][opp-j] = M[i+j][opp]

                M[i+j][opp] = temp
  1-02-2023(LEETCODE)
1)Binary Tree Maximum Path Sum   
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:

        self.ans = float('-inf')
        def leftRightSum(root):
            if not root:
                return 0
            left = max(leftRightSum(root.left), 0)
            right = max(leftRightSum(root.right), 0)
            self.ans = max(left+right+root.val, self.ans)
            return max(left, right) + root.val
        
        leftRightSum(root)
        return self.ans
 2)Valid Boomerang
 class Solution:
    def isBoomerang(self, points: List[List[int]]) -> bool:
        x1,y1 = points[0]
        x2,y2 = points[1]
        x3,y3 = points[2]

        area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        if area !=0:
            return True
        return False
 3)Binary Tree Tilt
 class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        sum = 0
        def dfs(root):
            if not root:
                return 0
            else:
                left = dfs(root.left)
                right = dfs(root.right)
                sum = abs(left - right)

            return left + root.val + right
        dfs(root)
        return sum
  4)Remove Duplicates from Sorted List
 class Solution:
    def deleteDuplicates(self, head):
        cur = head
        while cur != None:
            if cur.next == None:
                break
            else:
                if cur.val == cur.next.val:
                    cur.next = cur.next.next
                else:
                    cur = cur.next
        return head
 5)Add Binary
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a,2)+int(b,2))[2:]


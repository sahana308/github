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
 HACKERRANK(2-02--2023)
 1)Tree: Height of a Binary Tree
 class Node:
      def __init__(self,info): 
          self.info = info  
          self.left = None  
          self.right = None 

def height(root):
    l=0
    r=0
    if(root.left):
        l=height(root.left) + 1
    if(root.right):
        r=height(root.right) + 1
    if(l>r):
        return l
    else: 
        return r
 2)Binary Search Tree : Lowest Common Ancestor
 def lca(root, n1, n2):
    if (root.info < n1 and root.info > n2) or (root.info > n1 and root.info < n2):
        return root
    elif root.info < n1 and root.info < n2:
        return lca(root.right, n1, n2)
    elif root.info > n1 and root.info > n2:
        return lca(root.left, n1, n2)
    elif root.info == n1 or root.info == n2:
        return root
 3)Tree : Top View
 def topView(root):
    h={}
    queue=[]
    queue.append((root,0))
    while(queue):
        q=queue.pop(0)
        if q[1] not in h:
            h[q[1]]=q[0].info
        if q[0].left:
            queue.append((q[0].left,q[1]-1))
        if q[0].right:
            queue.append((q[0].right,q[1]+1))
    for k, v in sorted(h.items()):
        print(str(v)+' ', end='')
  4)Tree: Level Order Traversal
  def levelOrder(root):
    res = [root]
    
    while(len(res) > 0):
        a = res.pop(0)
        print(a.info,end = " ")
        if (a.left != None):
            res.append(a.left)
        if (a.right != None):
            res.append(a.right)
  LEETCODE(2-02-2023)
  1)Validate Binary Search Tree
  class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool :
        def dfs(node, low, high):
            if not node:
                return True
            if not (low < node.val < high):
                return False
            return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)
        
        return dfs(root, -inf, inf)
  2)Convert Sorted Array to Binary Search Tree
  class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        total = len(nums)
        if not total:
            return None

        mid_node = total // 2
        return TreeNode(
            nums[mid_node], 
            self.sortedArrayToBST(nums[:mid_node]), self.sortedArrayToBST(nums[mid_node + 1 :])
        )
  3)Convert Sorted List to Binary Search Tree
  class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
          if not head: return head
          if not head.next: return TreeNode(head.val)
        
          slow, fast = head, head.next.next
        
          while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
            tmp = slow.next
            slow.next = None
        
            root = TreeNode(tmp.val)
            root.left = self.sortedListToBST(head)
            root.right = self.sortedListToBST(tmp.next)
                return root
     4) All Elements in Two Binary Search Trees
     class Solution:
    def getAllElements(self, root1, root2):
        values = []
        def ct(root):
            if root:
                ct(root.left)
                values.append(root.val)
                ct(root.right)
        ct(root1)
        ct(root2)
        return sorted(values)
     5) Minimum Absolute Difference in BST
     class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        
        nodes = []
        
        def inr(root, nodes=[]):
            if not root: return None
            
            inr(root.left, nodes)
            nodes.append(root.val)
            inr(root.right, nodes)
        
        inr(root, nodes)
        
        return min(nodes[x+1] - nodes[x] for x in range(len(nodes) - 1))
    6)Construct Binary Search Tree from Preorder Traversal
    class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        root = TreeNode(preorder[0])
        s = [root]
        for value in preorder[1:]:
            if value < s[-1].val:
                s[-1].left = TreeNode(value)
                s.append(s[-1].left)
            else:
                while s and s[-1].val < value:
                    last = s.pop()
                last.right = TreeNode(value)
                s.append(last.right)
        return root
      7) Insert into a Binary Search Tree
      class Solution(object):
    def insertIntoBST(self, root, val):
        if root is None: return TreeNode(val)
        if val > root.val: root.right = self.insertIntoBST(root.right, val)  
        else: root.left = self.insertIntoBST(root.left, val)  
        return root

05-02-2023
import random
ladder = {3:24,35:55,64:95}
snake = {12:4,34:16,56:23}
p1=0
p2=0
def move(pos):
    dice = random.randint(1,6)
    print(f"Dice:{dice}")
    pos = pos + dice
    if pos in snake:
        print("Bitten by snake")
        pos = pos[snake]
        print(f"Position:{pos}")
    elif pos in ladder:
        print("Climbed by ladder")
        pos = pos[ladder]
        print(f"Position:{pos}")
    else:
         print(f"Position:{pos}")
    print("\n")
    return pos
while True:
    A = input("player 1 enter \"A\" to throw dice:")
    p1 = move(p1)
    if p1 >= 100:
        print("Game over!!!\n Player wins.")
        break
    B = input("player 2 enter \"B\" to throw dice:")
    p2 = move(pos2)
    if p2 >= 100:
        print("Game over!!!\n Player wins.")
        break
 07-02-23
 Number of Islands(LEETCODE)
 class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid: return 0

        isl = 0
        visited = set()

        for row in range(len(grid)):
            for col in range(len(grid[0])):
                isl += self.explore(grid, row, col, visited)
        
        return isl
    
    def explore(self, grid, row, col, visited):
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row]) or grid[row][col] == "0" or (row, col) in visited:
            return 0
        
        visited.add((row, col))

        self.explore(grid, row + 1, col, visited)
        self.explore(grid, row - 1, col, visited)
        self.explore(grid, row, col + 1, visited)
        self.explore(grid, row, col - 1, visited)

        return 1
 08-02-2023
 Rotting Oranges(LEETCODE)
 class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        fresh = 0 
        time = 0
        m,n = len(grid),len(grid[0])
        directions = [[0,1],[0,-1],[1,0],[-1,0]]
        q = collections.deque() 
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1:
                    fresh += 1
                elif grid[r][c] == 2:
                    q.append((r,c))
        
        while q and fresh > 0:
            for _ in range(len(q)):
                r,c = q.popleft()
                for d in directions:
                    row,col = r + d[0],c + d[1]
                    if (row >= 0 and row < m and col >= 0 and col < n
                         and grid[row][col] == 1):
                        fresh -= 1
                        grid[row][col] = 2
                        q.append((row,col))
                    
            time += 1
        
        return time if not fresh else -1
                                    
            

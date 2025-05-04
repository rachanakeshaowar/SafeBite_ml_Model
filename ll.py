import sys
from collections import defaultdict, deque
import heapq

# Increase recursion depth limit for deep trees
# The driver code already does this, but good practice to be aware
# sys.setrecursionlimit(1000005) # Typically set higher than N

class Solution:
    def minCost(self, n, roads, cost, k, journeys):
        
        MOD = 10**9 + 7

        # --- 1. Build Graph and Precompute Tree Properties ---
        adj = defaultdict(list)
        for u, v in roads:
            adj[u].append(v)
            adj[v].append(u)

        # Use 1-based indexing for cities consistently
        city_cost = [0] * (n + 1) # city_cost[i] is cost of city i
        for i in range(n):
            city_cost[i + 1] = cost[i]

        parent = [0] * (n + 1)
        path_cost = [0] * (n + 1) # Cost of path from city 1 to city i (inclusive)
        children = defaultdict(list) # Store children for post-order traversal
        
        q = deque([(1, 0)]) # (node, cost_so_far_exclusive_of_current)
        visited = {1}
        parent[1] = 0 # Root has no parent in this context
        path_cost[1] = city_cost[1]

        bfs_order = [] # To help determine processing order for post-order sum if needed

        while q:
            u, cost_so_far = q.popleft()
            bfs_order.append(u)
            
            current_node_cost = city_cost[u]
            current_path_cost = (cost_so_far + current_node_cost) % MOD
            # path_cost[u] = current_path_cost # This line was incorrect - path_cost is cumulative

            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    children[u].append(v) # u is parent of v
                    path_cost[v] = (path_cost[u] + city_cost[v]) % MOD # Correct path cost calculation
                    q.append((v, path_cost[u])) # Pass cumulative cost *up to parent*

        # --- 2. Group Journeys by Day ---
        journeys_by_day = defaultdict(list)
        for c, d in journeys:
            journeys_by_day[d].append(c)

        # --- 3. Process Each Day ---
        grand_total_cost = 0

        for day in journeys_by_day:
            destinations = journeys_by_day[day]
            
            # --- 3a. Calculate Initial Cost and Journeys Ending At Each Node ---
            current_day_initial_cost = 0
            journeys_ending_at = [0] * (n + 1) # Count journeys ending exactly at node i on this day
            
            for c in destinations:
                current_day_initial_cost = (current_day_initial_cost + path_cost[c]) % MOD
                journeys_ending_at[c] += 1

            # --- 3b. Calculate Subtree Journey Counts (Post-order traversal) ---
            # This count represents how many journeys on this day pass through node u
            subtree_journeys = [0] * (n + 1)

            # We can use the bfs_order in reverse for iterative post-order logic
            # Or use recursion (check recursion depth limit)
            
            # Recursive approach (ensure recursion depth is sufficient)
            memo = {} # Optional memoization if needed, but tree structure prevents cycles
            def dfs_post_order(u):
                if u in memo: return memo[u] # Should not be hit in a tree traversal from root
                
                count = journeys_ending_at[u]
                for v in children[u]:
                     count += dfs_post_order(v)
                subtree_journeys[u] = count
                # memo[u] = count # Store if needed, though not strictly necessary for tree
                return count

            # Start DFS from the root (city 1)
            dfs_post_order(1)
            
            # Iterative approach using reversed BFS order (guarantees children processed before parents)
            # for u in reversed(bfs_order):
            #     count = journeys_ending_at[u]
            #     for v in children[u]:
            #         count += subtree_journeys[v] # Child results are already computed
            #     subtree_journeys[u] = count


            # --- 3c. Calculate Potential Reductions ---
            potential_reductions = [] # Store (reduction_amount, city_index)
            for i in range(1, n + 1):
                # Reduction is 10% of the city cost, applied for each journey passing through it
                # Cost is guaranteed to be multiple of 10, so // 10 is exact 10%
                reduction = subtree_journeys[i] * (city_cost[i] // 10)
                if reduction > 0:
                    # Use negative reduction for max-heap behavior with min-heap library
                    heapq.heappush(potential_reductions, -reduction) 
                    # Alternatively: store positive values and sort later, or use a max-heap if available
                    # potential_reductions.append(reduction) 


            # --- 3d. Apply Top K Reductions ---
            total_day_reduction = 0
            coupons_used = 0
            
            # Using heapq (min-heap storing negative values to simulate max-heap)
            while coupons_used < k and potential_reductions:
                max_reduction = -heapq.heappop(potential_reductions) # Get largest reduction
                total_day_reduction = (total_day_reduction + max_reduction) % MOD
                coupons_used += 1

            # Alternative: Sorting
            # potential_reductions.sort(reverse=True)
            # for i in range(min(k, len(potential_reductions))):
            #     total_day_reduction = (total_day_reduction + potential_reductions[i]) % MOD


            # --- 3e. Calculate Final Cost for the Day ---
            # (a - b) % mod = (a - b + mod) % mod to handle potential negative results
            final_day_cost = (current_day_initial_cost - total_day_reduction + MOD) % MOD
            
            # --- 3f. Add to Grand Total ---
            grand_total_cost = (grand_total_cost + final_day_cost) % MOD

        # --- 4. Return Grand Total ---
        return grand_total_cost



#{ Driver Code Starts
#Initial Template for Python 3

import sys
sys.setrecursionlimit(1000005) # Set recursion depth limit

from collections import defaultdict, deque
import heapq # Import heapq for the optimized solution

if __name__ == "__main__":
    tt = int(sys.stdin.readline())
    for _ in range(tt):
        n = int(sys.stdin.readline())
        roads = []
        for _ in range(n - 1):
            u, v = map(int, sys.stdin.readline().split())
            roads.append([u, v])

        cost = list(map(int, sys.stdin.readline().split()))

        k = int(sys.stdin.readline())
        m = int(sys.stdin.readline())
        journeys = []
        for _ in range(m):
            c, d = map(int, sys.stdin.readline().split())
            journeys.append([c, d])

        ob = Solution()
        result = ob.minCost(n, roads, cost, k, journeys)
        print(result)
        # The problem statement output format seemed to have a "~" separator 
        # between test cases in some contexts, but standard output usually doesn't.
        # Let's stick to just printing the result unless the platform requires "~".
        # print("~") 
#} Driver Code Ends
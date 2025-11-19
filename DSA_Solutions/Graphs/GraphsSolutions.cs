// DSA Solutions - Graphs
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.Graphs
{
    // 1. Number of Islands
    public class NumberOfIslands
    {
        public int NumIslands(char[][] grid)
        {
            int m = grid.Length, n = grid[0].Length, count = 0;
            void Dfs(int i, int j)
            {
                if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') return;
                grid[i][j] = '0';
                Dfs(i + 1, j); Dfs(i - 1, j); Dfs(i, j + 1); Dfs(i, j - 1);
            }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (grid[i][j] == '1') { count++; Dfs(i, j); }
            return count;
        }
        // Explanation: DFS to mark all connected land.
    }

    // 2. Clone Graph
    public class CloneGraph
    {
        public Node CloneGraphMethod(Node node)
        {
            if (node == null) return null;
            var map = new Dictionary<Node, Node>();
            Node Dfs(Node n)
            {
                if (map.ContainsKey(n)) return map[n];
                var copy = new Node(n.val);
                map[n] = copy;
                foreach (var nei in n.neighbors) copy.neighbors.Add(Dfs(nei));
                return copy;
            }
            return Dfs(node);
        }
        public class Node
        {
            public int val;
            public IList<Node> neighbors;
            public Node(int _val) { val = _val; neighbors = new List<Node>(); }
        }
        // Explanation: DFS with hashmap to clone nodes.
    }

    // 3. Course Schedule
    public class CourseSchedule
    {
        public bool CanFinish(int numCourses, int[][] prerequisites)
        {
            var graph = new List<int>[numCourses];
            for (int i = 0; i < numCourses; i++) graph[i] = new List<int>();
            foreach (var pre in prerequisites) graph[pre[0]].Add(pre[1]);
            var visited = new int[numCourses];
            bool Dfs(int v)
            {
                if (visited[v] == 1) return false;
                if (visited[v] == 2) return true;
                visited[v] = 1;
                foreach (var nei in graph[v]) if (!Dfs(nei)) return false;
                visited[v] = 2;
                return true;
            }
            for (int i = 0; i < numCourses; i++) if (!Dfs(i)) return false;
            return true;
        }
        // Explanation: DFS cycle detection for topological sort.
    }

    // 4. Pacific Atlantic Water Flow
    public class PacificAtlanticWaterFlow
    {
        public IList<IList<int>> PacificAtlantic(int[][] heights)
        {
            int m = heights.Length, n = heights[0].Length;
            var pacific = new bool[m, n];
            var atlantic = new bool[m, n];
            var res = new List<IList<int>>();
            void Dfs(int i, int j, bool[,] ocean, int prev)
            {
                if (i < 0 || i >= m || j < 0 || j >= n || ocean[i, j] || heights[i][j] < prev) return;
                ocean[i, j] = true;
                Dfs(i + 1, j, ocean, heights[i][j]);
                Dfs(i - 1, j, ocean, heights[i][j]);
                Dfs(i, j + 1, ocean, heights[i][j]);
                Dfs(i, j - 1, ocean, heights[i][j]);
            }
            for (int i = 0; i < m; i++) { Dfs(i, 0, pacific, int.MinValue); Dfs(i, n - 1, atlantic, int.MinValue); }
            for (int j = 0; j < n; j++) { Dfs(0, j, pacific, int.MinValue); Dfs(m - 1, j, atlantic, int.MinValue); }
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (pacific[i, j] && atlantic[i, j]) res.Add(new List<int> { i, j });
            return res;
        }
        // Explanation: DFS from ocean borders, intersection is answer.
    }

    // 5. Rotting Oranges
    public class RottingOranges
    {
        public int OrangesRotting(int[][] grid)
        {
            int m = grid.Length, n = grid[0].Length, fresh = 0, minutes = 0;
            var queue = new Queue<(int, int)>();
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    if (grid[i][j] == 2) queue.Enqueue((i, j));
                    else if (grid[i][j] == 1) fresh++;
            int[][] dirs = { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 0, -1 }, new[] { -1, 0 } };
            while (queue.Count > 0 && fresh > 0)
            {
                int size = queue.Count;
                for (int i = 0; i < size; i++)
                {
                    var (x, y) = queue.Dequeue();
                    foreach (var d in dirs)
                    {
                        int nx = x + d[0], ny = y + d[1];
                        if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1)
                        {
                            grid[nx][ny] = 2;
                            queue.Enqueue((nx, ny));
                            fresh--;
                        }
                    }
                }
                minutes++;
            }
            return fresh == 0 ? minutes : -1;
        }
        // Explanation: BFS to spread rot, count minutes.
    }

    // 6. Word Ladder
    public class WordLadder
    {
        public int LadderLength(string beginWord, string endWord, IList<string> wordList)
        {
            var set = new HashSet<string>(wordList);
            if (!set.Contains(endWord)) return 0;
            var queue = new Queue<(string, int)>();
            queue.Enqueue((beginWord, 1));
            while (queue.Count > 0)
            {
                var (word, len) = queue.Dequeue();
                if (word == endWord) return len;
                for (int i = 0; i < word.Length; i++)
                {
                    var arr = word.ToCharArray();
                    for (char c = 'a'; c <= 'z'; c++)
                    {
                        arr[i] = c;
                        var next = new string(arr);
                        if (set.Contains(next))
                        {
                            queue.Enqueue((next, len + 1));
                            set.Remove(next);
                        }
                    }
                }
            }
            return 0;
        }
        // Explanation: BFS, try all one-letter transformations.
    }

    // 7. Graph Valid Tree
    public class GraphValidTree
    {
        public bool ValidTree(int n, int[][] edges)
        {
            if (edges.Length != n - 1) return false;
            var parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
            int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
            foreach (var e in edges)
            {
                int x = Find(e[0]), y = Find(e[1]);
                if (x == y) return false;
                parent[x] = y;
            }
            return true;
        }
        // Explanation: Union-find to check for cycles and connectivity.
    }

    // 8. Network Delay Time
    public class NetworkDelayTime
    {
        public int NetworkDelayTimeMethod(int[][] times, int n, int k)
        {
            var graph = new List<(int, int)>[n + 1];
            for (int i = 1; i <= n; i++) graph[i] = new List<(int, int)>();
            foreach (var t in times) graph[t[0]].Add((t[1], t[2]));
            var dist = new int[n + 1];
            Array.Fill(dist, int.MaxValue);
            dist[k] = 0;
            var pq = new SortedSet<(int, int)>();
            pq.Add((0, k));
            while (pq.Count > 0)
            {
                var (d, u) = pq.Min; pq.Remove(pq.Min);
                foreach (var (v, w) in graph[u])
                {
                    if (dist[v] > d + w)
                    {
                        pq.Remove((dist[v], v));
                        dist[v] = d + w;
                        pq.Add((dist[v], v));
                    }
                }
            }
            int ans = 0;
            for (int i = 1; i <= n; i++)
                if (dist[i] == int.MaxValue) return -1;
                else ans = Math.Max(ans, dist[i]);
            return ans;
        }
        // Explanation: Dijkstra's algorithm for shortest path.
    }

    // 9. Find Eventual Safe States
    public class FindEventualSafeStates
    {
        public IList<int> EventualSafeNodes(int[][] graph)
        {
            int n = graph.Length;
            var color = new int[n];
            var res = new List<int>();
            bool Dfs(int node)
            {
                if (color[node] > 0) return color[node] == 2;
                color[node] = 1;
                foreach (var nei in graph[node])
                    if (!Dfs(nei)) return false;
                color[node] = 2;
                return true;
            }
            for (int i = 0; i < n; i++) if (Dfs(i)) res.Add(i);
            return res;
        }
        // Explanation: DFS with coloring to find safe nodes.
    }

    // 10. Number of Connected Components
    public class NumberOfConnectedComponents
    {
        public int CountComponents(int n, int[][] edges)
        {
            var parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
            int Find(int x) { return parent[x] == x ? x : parent[x] = Find(parent[x]); }
            foreach (var e in edges)
            {
                int x = Find(e[0]), y = Find(e[1]);
                if (x != y) parent[x] = y;
            }
            var set = new HashSet<int>();
            for (int i = 0; i < n; i++) set.Add(Find(i));
            return set.Count;
        }
        // Explanation: Union-find to count connected components.
    }
}

# Recursion & Backtracking DSA Interview Solutions

This document contains C# solutions, explanations, and time/space complexity analysis for common recursion and backtracking DSA interview questions.

---

## 1. Subsets
**Problem:** Generate all possible subsets.

```csharp
public IList<IList<int>> SubsetsMethod(int[] nums)
{
    var res = new List<IList<int>>();
    Backtrack(0, new List<int>());
    return res;
    void Backtrack(int start, List<int> curr)
    {
        res.Add(new List<int>(curr));
        for (int i = start; i < nums.Length; i++)
        {
            curr.Add(nums[i]);
            Backtrack(i + 1, curr);
            curr.RemoveAt(curr.Count - 1);
        }
    }
}
```
**Explanation:** Backtracking to generate all subsets.
**Time Complexity:** O(2^n)
**Space Complexity:** O(n * 2^n)

**Dry Run Example:**
Input: [1,2]
Backtrack(0, []): add []
Add 1, Backtrack(1, [1]): add [1]
Add 2, Backtrack(2, [1,2]): add [1,2], remove 2
Remove 1
Add 2, Backtrack(2, [2]): add [2], remove 2
Output: [[], [1], [1,2], [2]]

---

## 2. Permutations
**Problem:** Generate all possible permutations.

```csharp
public IList<IList<int>> Permute(int[] nums)
{
    var res = new List<IList<int>>();
    Backtrack(new List<int>());
    return res;
    void Backtrack(List<int> curr)
    {
        if (curr.Count == nums.Length)
        {
            res.Add(new List<int>(curr));
            return;
        }
        for (int i = 0; i < nums.Length; i++)
        {
            if (curr.Contains(nums[i])) continue;
            curr.Add(nums[i]);
            Backtrack(curr);
            curr.RemoveAt(curr.Count - 1);
        }
    }
}
```
**Explanation:** Backtracking to generate all permutations.
**Time Complexity:** O(n!)
**Space Complexity:** O(n * n!)

**Dry Run Example:**
Input: [1,2,3]
Backtrack([]): add 1, Backtrack([1]): add 2, Backtrack([1,2]): add 3, Backtrack([1,2,3]): add [1,2,3]
Remove 3, remove 2, add 3, Backtrack([1,3]): add 2, Backtrack([1,3,2]): add [1,3,2]
... (all permutations)
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

---

## 3. Combination Sum
**Problem:** Find all combinations that sum to target.

```csharp
public IList<IList<int>> CombinationSumMethod(int[] candidates, int target)
{
    var res = new List<IList<int>>();
    Backtrack(0, target, new List<int>());
    return res;
    void Backtrack(int start, int remain, List<int> curr)
    {
        if (remain == 0)
        {
            res.Add(new List<int>(curr));
            return;
        }
        if (remain < 0) return;
        for (int i = start; i < candidates.Length; i++)
        {
            curr.Add(candidates[i]);
            Backtrack(i, remain - candidates[i], curr);
            curr.RemoveAt(curr.Count - 1);
        }
    }
}
```
**Explanation:** Backtracking to find combinations that sum to target.
**Time Complexity:** O(2^n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: candidates=[2,3,6,7], target=7
Backtrack(0,7,[]): add 2, Backtrack(0,5,[2]): add 2, Backtrack(0,3,[2,2]): add 2, Backtrack(0,1,[2,2,2]): add 2, Backtrack(0,-1,[2,2,2,2]): return
Remove 2, add 3, Backtrack(1,-2,[2,2,3]): return
... (eventually finds [7])
Output: [[2,2,3],[7]]

---

## 4. Combination Sum II
**Problem:** Find all unique combinations that sum to target (no repeats).

```csharp
public IList<IList<int>> CombinationSum2(int[] candidates, int target)
{
    Array.Sort(candidates);
    var res = new List<IList<int>>();
    Backtrack(0, target, new List<int>());
    return res;
    void Backtrack(int start, int remain, List<int> curr)
    {
        if (remain == 0)
        {
            res.Add(new List<int>(curr));
            return;
        }
        if (remain < 0) return;
        for (int i = start; i < candidates.Length; i++)
        {
            if (i > start && candidates[i] == candidates[i - 1]) continue;
            curr.Add(candidates[i]);
            Backtrack(i + 1, remain - candidates[i], curr);
            curr.RemoveAt(curr.Count - 1);
        }
    }
}
```
**Explanation:** Backtracking with duplicate skipping.
**Time Complexity:** O(2^n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: candidates=[10,1,2,7,6,1,5], target=8
Sort: [1,1,2,5,6,7,10]
Backtrack(0,8,[]): add 1, Backtrack(1,7,[1]): add 1, Backtrack(2,6,[1,1]): ...
Skip duplicates, only unique combinations
Output: [[1,1,6],[1,2,5],[1,7],[2,6]]

---

## 5. Letter Combinations of a Phone Number
**Problem:** Generate all possible letter combinations for a phone number.

```csharp
private static readonly string[] map = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
public IList<string> LetterCombinations(string digits)
{
    var res = new List<string>();
    if (string.IsNullOrEmpty(digits)) return res;
    Backtrack(0, "");
    return res;
    void Backtrack(int index, string curr)
    {
        if (index == digits.Length)
        {
            res.Add(curr);
            return;
        }
        foreach (var c in map[digits[index] - '0'])
        {
            Backtrack(index + 1, curr + c);
        }
    }
}
```
**Explanation:** Backtracking to generate all letter combinations.
**Time Complexity:** O(4^n)
**Space Complexity:** O(n * 4^n)

**Dry Run Example:**
Input: digits="23"
Backtrack(0,""): add 'a', Backtrack(1,"a"): add 'd', Backtrack(2,"ad"): add "ad"
... (all combinations)
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

---

## 6. N-Queens
**Problem:** Solve the N-Queens puzzle.

```csharp
public IList<IList<string>> SolveNQueens(int n)
{
    var res = new List<IList<string>>();
    var board = new char[n, n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            board[i, j] = '.';
    Backtrack(0);
    return res;
    void Backtrack(int row)
    {
        if (row == n)
        {
            var solution = new List<string>();
            for (int i = 0; i < n; i++)
            {
                var sb = new System.Text.StringBuilder();
                for (int j = 0; j < n; j++) sb.Append(board[i, j]);
                solution.Add(sb.ToString());
            }
            res.Add(solution);
            return;
        }
        for (int col = 0; col < n; col++)
        {
            if (IsValid(row, col))
            {
                board[row, col] = 'Q';
                Backtrack(row + 1);
                board[row, col] = '.';
            }
        }
    }
    bool IsValid(int row, int col)
    {
        for (int i = 0; i < row; i++)
            if (board[i, col] == 'Q') return false;
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
            if (board[i, j] == 'Q') return false;
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++)
            if (board[i, j] == 'Q') return false;
        return true;
    }
}
```
**Explanation:** Backtracking to place queens row by row.
**Time Complexity:** O(n!)
**Space Complexity:** O(n^2)

**Dry Run Example:**
Input: n=4
Backtrack(0): try col=0, place Q, Backtrack(1): try col=2, place Q, ...
Finds all valid board configurations
Output: 2 solutions for n=4

---

## 7. Word Search
**Problem:** Search for a word in a grid.

```csharp
public bool Exist(char[][] board, string word)
{
    int m = board.Length, n = board[0].Length;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (Backtrack(i, j, 0)) return true;
    return false;
    bool Backtrack(int i, int j, int k)
    {
        if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != word[k]) return false;
        if (k == word.Length - 1) return true;
        char temp = board[i][j];
        board[i][j] = '#';
        bool found = Backtrack(i + 1, j, k + 1) || Backtrack(i - 1, j, k + 1) || Backtrack(i, j + 1, k + 1) || Backtrack(i, j - 1, k + 1);
        board[i][j] = temp;
        return found;
    }
}
```
**Explanation:** Backtracking to search for word in grid.
**Time Complexity:** O(mn * 4^L)
**Space Complexity:** O(L)

**Dry Run Example:**
Input: board=[[A,B,C,E],[S,F,C,S],[A,D,E,E]], word="ABCCED"
Start at (0,0): A, Backtrack to (0,1): B, (0,2): C, (1,2): C, (1,1): E, (2,1): D
Output: true

---

## 8. Palindrome Partitioning
**Problem:** Partition a string into all possible palindrome partitions.

```csharp
public IList<IList<string>> Partition(string s)
{
    var res = new List<IList<string>>();
    Backtrack(0, new List<string>());
    return res;
    void Backtrack(int start, List<string> curr)
    {
        if (start == s.Length)
        {
            res.Add(new List<string>(curr));
            return;
        }
        for (int end = start + 1; end <= s.Length; end++)
        {
            var substr = s.Substring(start, end - start);
            if (IsPalindrome(substr))
            {
                curr.Add(substr);
                Backtrack(end, curr);
                curr.RemoveAt(curr.Count - 1);
            }
        }
    }
    bool IsPalindrome(string str)
    {
        int l = 0, r = str.Length - 1;
        while (l < r)
        {
            if (str[l++] != str[r--]) return false;
        }
        return true;
    }
}
```
**Explanation:** Backtracking to partition string into palindromes.
**Time Complexity:** O(n * 2^n)
**Space Complexity:** O(n^2)

**Dry Run Example:**
Input: s="aab"
Backtrack(0,[]): substr="a" (palindrome), add, Backtrack(1,["a"]): substr="a" (palindrome), add, Backtrack(2,["a","a"]): substr="b" (palindrome), add, Backtrack(3,["a","a","b"]): add to res
... (all partitions)
Output: [["a","a","b"],["aa","b"]]

---

## 9. Generate Parentheses
**Problem:** Generate all valid parentheses combinations.

```csharp
public IList<string> GenerateParenthesis(int n)
{
    var res = new List<string>();
    Backtrack("", 0, 0);
    return res;
    void Backtrack(string curr, int open, int close)
    {
        if (curr.Length == 2 * n)
        {
            res.Add(curr);
            return;
        }
        if (open < n) Backtrack(curr + "(", open + 1, close);
        if (close < open) Backtrack(curr + ")", open, close + 1);
    }
}
```
**Explanation:** Backtracking to generate all valid parentheses.
**Time Complexity:** O(2^n)
**Space Complexity:** O(n)

**Dry Run Example:**
Input: n=3
Backtrack("",0,0): add '(', Backtrack("(",1,0): add '(', Backtrack("((",2,0): add '(', Backtrack("(((",3,0): add ')', ...
Output: ["((()))","(()())","(())()","()(())","()()()"]

---

## 10. Sudoku Solver
**Problem:** Solve a Sudoku puzzle.

```csharp
public void SolveSudoku(char[][] board)
{
    Backtrack();
    void Backtrack()
    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (board[i][j] == '.')
                {
                    for (char c = '1'; c <= '9'; c++)
                    {
                        if (IsValid(i, j, c))
                        {
                            board[i][j] = c;
                            Backtrack();
                            if (IsSolved()) return;
                            board[i][j] = '.';
                        }
                    }
                    return;
                }
            }
        }
    }
    bool IsValid(int row, int col, char c)
    {
        for (int i = 0; i < 9; i++)
        {
            if (board[row][i] == c || board[i][col] == c) return false;
            if (board[3 * (row / 3) + i / 3][3 * (col / 3) + i % 3] == c) return false;
        }
        return true;
    }
    bool IsSolved()
    {
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                if (board[i][j] == '.') return false;
        return true;
    }
}
```
**Explanation:** Backtracking to fill sudoku board.
**Time Complexity:** O(9^(n^2))
**Space Complexity:** O(n^2)

**Dry Run Example:**
Input: board partially filled
Backtrack(): try each empty cell, try 1-9, check IsValid, fill and recurse
Backtrack on wrong guess, fill correct value
Output: solved board

---

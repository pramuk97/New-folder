// DSA Solutions - Recursion & Backtracking
// Author: 12 years C# experience
// Each solution includes code and explanation for interview preparation

using System;
using System.Collections.Generic;

namespace DSA_Solutions.RecursionBacktracking
{
    // 1. Subsets
    public class Subsets
    {
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
        // Explanation: Backtracking to generate all subsets.
    }

    // 2. Permutations
    public class Permutations
    {
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
        // Explanation: Backtracking to generate all permutations.
    }

    // 3. Combination Sum
    public class CombinationSum
    {
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
        // Explanation: Backtracking to find combinations that sum to target.
    }

    // 4. Combination Sum II
    public class CombinationSumII
    {
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
        // Explanation: Backtracking with duplicate skipping.
    }

    // 5. Letter Combinations of a Phone Number
    public class LetterCombinationsOfPhoneNumber
    {
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
        // Explanation: Backtracking to generate all letter combinations.
    }

    // 6. N-Queens
    public class NQueens
    {
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
        // Explanation: Backtracking to place queens row by row.
    }

    // 7. Word Search
    public class WordSearch
    {
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
        // Explanation: Backtracking to search for word in grid.
    }

    // 8. Palindrome Partitioning
    public class PalindromePartitioning
    {
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
        // Explanation: Backtracking to partition string into palindromes.
    }

    // 9. Generate Parentheses
    public class GenerateParentheses
    {
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
        // Explanation: Backtracking to generate all valid parentheses.
    }

    // 10. Sudoku Solver
    public class SudokuSolver
    {
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
        // Explanation: Backtracking to fill sudoku board.
    }
}

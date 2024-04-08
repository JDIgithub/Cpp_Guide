#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>


bool hasPairWithSum(const std::vector<int>& numbers, int targetSum) {

  int left = 0; // Start from the beginning of the array
  int right = numbers.size() - 1; // Start from the end of the array

  while (left < right) {
    int currentSum = numbers[left] + numbers[right];

    if (currentSum == targetSum) {
      return true; // Found a pair
    } else if (currentSum < targetSum) {
      left++; // Increase the sum
    } else {
      right--; // Decrease the sum
    }
  }
  return false; // No pair found
}

int main() {
  std::vector<int> sortedNumbers = {1, 2, 4, 4, 5, 6, 8, 9};
  int target = 8;

  if (hasPairWithSum(sortedNumbers, target)) {
    std::cout << "Pair with the given sum exists." << std::endl;
  } else {
    std::cout << "No pair with the given sum exists." << std::endl;
  }

  return 0;
}




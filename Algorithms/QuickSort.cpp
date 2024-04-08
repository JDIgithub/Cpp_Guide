#include <iostream>
#include <vector>


// Function to choose pivot and swap all elements smaller to the left of pivot 
int pivot(std::vector<int>& array, int pivotIndex, int endIndex) {
  
  int swapIndex = pivotIndex;

  for (int i = pivotIndex + 1; i <= endIndex; i++) {
    if(array[i] < array[pivotIndex]){
      swapIndex++;
      std::swap(array[i],array[swapIndex]);
    }
  }
  std::swap(array[pivotIndex], array[swapIndex]);
  return swapIndex;
}

// Function to perform QuickSort
void quickSort(std::vector<int>& array, int leftIndex, int rightIndex) {
  if (leftIndex >= rightIndex) { return; } // End of the recursion
  
  int pivotIndex = pivot(array, leftIndex, rightIndex);
  // Separately sorts sub-arrays to the left of the pivot and to the right of the pivot
  quickSort(array, leftIndex, pivotIndex - 1);
  quickSort(array, pivotIndex + 1, rightIndex);
}

int main() {
  
  std::vector<int> array = {3, 6, 8, 10, 1, 2, 1};
  int pivotIndex = 0;
  int rightIndex = array.size() - 1;
  quickSort(array, pivotIndex, rightIndex);

  std::cout << "Sorted array: \n";
  for (int element : array) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
  return 0;
}
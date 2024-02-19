#include <iostream>
#include <vector>

void quickSort(std::vector<int>& arr, int low, int high);
int partition(std::vector<int>& arr, int low, int high);
void swap(int* a, int* b);

int main() {
  
  std::vector<int> array = {3, 6, 8, 10, 1, 2, 1};
  int n = array.size();
  int low = 0;
  int high = n - 1;
  quickSort(array, low, high);

  std::cout << "Sorted array: \n";
  for (int i = 0; i < n; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}

// Function to perform QuickSort
void quickSort(std::vector<int>& array, int low, int high) {
  if (low < high) {
    int partitioningIndex = partition(array, low, high);

    // Separately sort elements before partition and after partition
    quickSort(array, low, partitioningIndex - 1);
    quickSort(array, partitioningIndex + 1, high);
  }
}

// Function to partition the array
int partition(std::vector<int>& array, int low, int high) {
  
  int pivot = array[high]; // pivot
  int i = (low - 1); // Index of smaller element

  for (int j = low; j <= high - 1; j++) {
    // If current element is smaller than the pivot
    if (array[j] < pivot) {
      i++; // increment index of smaller element
      swap(&array[i], &array[j]);
     }
  }
  swap(&array[i + 1], &array[high]);
  return (i + 1);
}

// Function to swap two elements
void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

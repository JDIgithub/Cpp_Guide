#include <iostream>

// Function to perform Insertion Sort
void insertionSort(int array[], int n) {
  
  int i, key, j;
  for (i = 1; i < n; i++) {
    key = array[i];
    j = i - 1;

    // Move elements of array[0..i-1], that are greater than key,
    // to one position ahead of their current position
    while (j >= 0 && array[j] > key) {
      array[j + 1] = array[j];
      j = j - 1;
    }
    array[j + 1] = key;
  }
}

// Main function
int main() {
  
    int array[] = {12, 11, 13, 5, 6};
    int n = sizeof(array) / sizeof(array[0]);
    insertionSort(array, n);

    return 0;
}





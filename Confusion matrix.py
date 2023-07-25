def calculate_accuracy(confusion_matrix):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]

    predictions = tp + fp + fn + tn
    accuracy = (tp + tn) / predictions

    return accuracy

def sort_confusion_matrices_by_accuracy(cmatrix):
    accuracy_with_matrix = [(calculate_accuracy(matrix), matrix) for matrix in cmatrix]
    sorted_matrices = sorted(accuracy_with_matrix, key=lambda x: x[0], reverse=True)
    sorted_matrices_only = [matrix[1] for matrix in sorted_matrices]
    return sorted_matrices_only

n=int(input("Enter the number of confusion matrixes:"))
cmatrix = []
for i in range(0,n):
    print(f"Enter the matrix {i+1}:")
    cmatrix.append([[int(input()),int(input())],[int(input()),int(input())]])
    
sorted_matrices = sort_confusion_matrices_by_accuracy(cmatrix)

acc=calculate_accuracy(sorted_matrices[0])
print(f"The Best Model based on Accuracy {acc:.2f} is:")
print()
print(f"\t{sorted_matrices[0][0][0]}\t{sorted_matrices[0][0][1]}")
print()
print(f"\t{sorted_matrices[0][1][0]}\t{sorted_matrices[0][1][1]}")
print()



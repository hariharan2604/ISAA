def cal_acc(confusion_matrix):
    tp=confusion_matrix[0][0]
    tn=confusion_matrix[1][1]
    fn=confusion_matrix[1][0]
    fp=confusion_matrix[1][1]
    acc=(tn+tp)/(tn+tp+fn+fp)
    return acc

def sorted_confusion_acc(cmatrix):
    with_acc=[(cal_acc(matrix),matrix) for matrix in cmatrix]
    sorted_matrix=sorted(with_acc, key=lambda x:x[0], reverse=True)
    sorted_matrix_only=[matrix[1] for matrix in sorted_matrix]
    return sorted_matrix_only

n=int(input("Enter the no of confusion matrix :"))
cmatrix=[]
for i in range(0,n):
    print(f"enter the matrix {i+1}:")
    cmatrix=([[int(input()),int(input())],[int(input()),int(input())]])
sorted=sorted_confusion_acc(cmatrix)
acc=cal_acc(sorted[0])
print(f"The best model on accuracy {acc.2f} ")
print()
print(f"\t{sorted[0][0][0]}\t{sorted[0][0][1]}")
print()
print(f"\t{sorted[0][1][0]}\t{sorted[0][1][1]}")

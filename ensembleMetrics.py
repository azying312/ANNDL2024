# Confusion matrix
conf_matrix = confusion_matrix(y_val_binary, ensemble_predictions_val)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=['No Recurrence', 'Recurrence'],
            yticklabels=['No Recurrence', 'Recurrence'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

### Ensemble Eval Metrics

# Accuracy
accuracy = accuracy_score(ensemble_predictions_val, y_val_binary)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(ensemble_predictions_val, y_val_binary)
print("Precision:", precision)

# Recall
recall = recall_score(ensemble_predictions_val, y_val_binary)
print("Recall:", recall)

# F1 Score
f1 = f1_score(ensemble_predictions_val, ensemble_predictions_val)
print("F1 Score:", f1)

# ROC AUC Score
# roc_auc = roc_auc_score(ensemble_predictions_val, y_val_binary)
# print("ROC AUC Score:", roc_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(y_val_binary, ensemble_predictions_val)
print("Confusion Matrix:")
print(conf_matrix)

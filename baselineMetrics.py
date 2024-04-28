# Make predictions on the test data
predictions = model.predict([baseline_mri_X_test, baseline_dose_X_test, baseline_X_test_processed])
predictions_binary = (predictions > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(baseline_y_test_binary, predictions_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=['No Recurrence', 'Recurrence'],
            yticklabels=['No Recurrence', 'Recurrence'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

### Baseline Eval Metrics

y_pred = model.predict([baseline_mri_X_test, baseline_dose_X_test, baseline_X_test_processed])
y_pred_binary = (y_pred > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(baseline_y_test_binary, y_pred_binary)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(baseline_y_test_binary, y_pred_binary)
print("Precision:", precision)

# Recall
recall = recall_score(baseline_y_test_binary, y_pred_binary)
print("Recall:", recall)

# F1 Score
f1 = f1_score(baseline_y_test_binary, y_pred_binary)
print("F1 Score:", f1)

# ROC AUC Score
roc_auc = roc_auc_score(baseline_y_test_binary, y_pred)
print("ROC AUC Score:", roc_auc)

# Confusion Matrix
conf_matrix = confusion_matrix(baseline_y_test_binary, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)

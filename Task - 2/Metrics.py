import matplotlib.pyplot as plt, numpy as np, os

os.makedirs("results/metrics", exist_ok=True)

# ---- Simulated confidence & ground truth values ----
conf = np.linspace(0, 1, 50)
prec = 0.9 - 0.4*conf + np.random.uniform(-0.02, 0.02, 50)
rec  = 0.5 + 0.4*conf + np.random.uniform(-0.02, 0.02, 50)
f1   = 2*(prec*rec)/(prec+rec)

# ---- 1. Precision–Confidence ----
plt.plot(conf, prec, 'b', lw=2); plt.title("Precision–Confidence Curve")
plt.xlabel("Confidence"); plt.ylabel("Precision"); plt.grid(True)
plt.savefig("results/metrics/precision_conf_curve.png"); plt.close()

# ---- 2. Recall–Confidence ----
plt.plot(conf, rec, 'g', lw=2); plt.title("Recall–Confidence Curve")
plt.xlabel("Confidence"); plt.ylabel("Recall"); plt.grid(True)
plt.savefig("results/metrics/recall_conf_curve.png"); plt.close()

# ---- 3. F1–Confidence ----
plt.plot(conf, f1, 'r', lw=2); plt.title("F1–Confidence Curve")
plt.xlabel("Confidence"); plt.ylabel("F1 Score"); plt.grid(True)
plt.savefig("results/metrics/f1_conf_curve.png"); plt.close()

# ---- 4. Precision–Recall ----
plt.plot(rec, prec, 'purple', lw=2); plt.title("Precision–Recall Curve")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True)
plt.savefig("results/metrics/precision_recall_curve.png"); plt.close()

# ---- 5. Confusion Matrix ----
y_true = np.array([1,0,1,1,0,1,0,0,1,1])
y_pred = np.array([1,0,1,0,0,1,1,0,1,0])
TP, TN = np.sum((y_true==1)&(y_pred==1)), np.sum((y_true==0)&(y_pred==0))
FP, FN = np.sum((y_true==0)&(y_pred==1)), np.sum((y_true==1)&(y_pred==0))
cm = np.array([[TN,FP],[FN,TP]])
plt.imshow(cm,cmap='pink'); plt.title("Confusion Matrix")
for i in range(2):
  for j in range(2): plt.text(j,i,cm[i,j],ha='center',va='center',color='black')
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.colorbar()
plt.savefig("results/metrics/confusion_matrix.png"); plt.close()

print(" All graphs saved in results/metrics folder.")

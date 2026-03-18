import numpy as np
import time
from collections import deque
 
class HorizontalPodAutoscaler:
    def __init__(self, min_replicas=1, max_replicas=10, target_cpu=70.0, cooldown=60):
        self.replicas = 2; self.min_r = min_replicas; self.max_r = max_replicas
        self.target = target_cpu; self.cooldown = cooldown; self.last_scale = -cooldown
 
    def desired_replicas(self, current_cpu, t):
        desired = int(np.ceil(self.replicas * current_cpu / self.target))
        desired = np.clip(desired, self.min_r, self.max_r)
        if abs(desired - self.replicas) > 0 and (t - self.last_scale) >= self.cooldown:
            self.replicas = desired; self.last_scale = t
        return self.replicas
 
class PredictiveScaler:
    """LSTM-based predictive autoscaler (simulation)."""
    def __init__(self, window=20):
        self.history = deque(maxlen=window); self.replicas = 2
    def add_metric(self, cpu): self.history.append(cpu)
    def predict_next(self):
        if len(self.history) < 5: return np.mean(self.history) if self.history else 50
        h = np.array(self.history)
        trend = np.polyfit(range(len(h)), h, 1)[0]
        return h[-1] + trend * 5  # predict 5 steps ahead
    def scale(self, target_cpu=70):
        predicted = self.predict_next()
        desired = int(np.ceil(self.replicas * predicted / target_cpu))
        self.replicas = int(np.clip(desired, 1, 10))
        return self.replicas
 
def simulate_workload(steps=300, pattern='diurnal'):
    t = np.arange(steps)
    if pattern == 'diurnal':
        return 30 + 50*np.sin(2*np.pi*t/200)**2 + np.random.randn(steps)*5
    return 50 + np.random.randn(steps)*15
 
workload = simulate_workload(300)
hpa = HorizontalPodAutoscaler()
pred_scaler = PredictiveScaler()
hpa_replicas = []; pred_replicas = []
for t, cpu in enumerate(workload):
    r1 = hpa.desired_replicas(cpu, t)
    pred_scaler.add_metric(cpu); r2 = pred_scaler.scale()
    hpa_replicas.append(r1); pred_replicas.append(r2)
  print(f"HPA  - mean replicas: {np.mean(hpa_replicas):.1f}, max: {max(hpa_replicas)}")
print(f"Pred - mean replicas: {np.mean(pred_replicas):.1f}, max: {max(pred_replicas)}")
print(f"Scale events (HPA): {sum(1 for i in range(1,len(hpa_replicas)) if hpa_replicas[i]!=hpa_replicas[i-1])}")

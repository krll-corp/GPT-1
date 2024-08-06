import numpy as np

times = [47909.32, 49319.51, 49183.43]
avg_time_per_iter = np.mean(times)

total_iters = 600000
total_time_ms = avg_time_per_iter * total_iters
total_time_hours = total_time_ms / (1000 * 60 * 60)
total_time_days = total_time_hours / 24

print(f"Среднее время на итерацию: {avg_time_per_iter:.2f} мс")
print(f"Примерное общее время обучения:")
print(f"{total_time_hours:.2f} часов")
print(f"{total_time_days:.2f} дней")
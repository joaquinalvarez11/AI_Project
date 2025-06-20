# recommendation_system.py
import numpy as np

class SimpleRecommender:
	def __init__(self, user_item_matrix):
		"""
		user_item_matrix: numpy array (users x items) with ratings or interactions
		"""
		self.user_item_matrix = user_item_matrix

	def recommend(self, user_index, top_n=3):
		"""
		Recommend top_n items for a given user based on average item scores.
		"""
		user_ratings = self.user_item_matrix[user_index]
		unrated_items = np.where(user_ratings == 0)[0]
		item_scores = self.user_item_matrix[:, unrated_items].mean(axis=0)
		top_items = unrated_items[np.argsort(item_scores)[::-1][:top_n]]
		return top_items

# Ejemplo de uso
if __name__ == "__main__":
	# Matriz de ejemplo:
	# filas = usuarios
	# columnas = items
	# valores = puntuaciones (0 = no evaluado)
	user_item_matrix = np.array([
		[5, 3, 0, 1],
		[4, 0, 0, 1],
		[1, 1, 0, 5],
		[0, 0, 5, 4],
		[0, 1, 5, 4],
	])

	recommender = SimpleRecommender(user_item_matrix)
	user_id = 0
	recomendaciones = recommender.recommend(user_id, top_n=2)
	print(f"Recomendaciones para el usuario {user_id}: {recomendaciones}")

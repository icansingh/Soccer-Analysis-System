'''
The problem with the team assigner is that it sometimes changes the team color for the same player based on
the jersey design. For example, if the jersey is red and the front is mostly red while the back is mostly
white (because of the player number and name), then when the player faces the camera, the team color is
assigned as red and when the player faces away from the camera, the team color is assigned as white.

This probably happens because I only take the top half of the image - update, I still get the same issue
even if I take the whole image. I think the issue is that the player's jersey is not a solid color.
'''

from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, top_half_image):
        # Reshape image to 2d array
        image_2d = top_half_image.reshape(-1, 3)

        # Perform KMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init = 1).fit(image_2d)  # Check notebook. k-means++ is faster

        return kmeans

    def get_player_color(self, frame, bounding_box):
        image = frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

        # Get top half of image
        #top_half = image[0 : image.shape[0] // 2, :]

        # Clustering model
        kmeans = self.get_clustering_model(image)

        # Get cluster labels
        labels = kmeans.labels_

        # Reshape labels back to the image shape
        clustered_image = labels.reshape(image.shape[0], image.shape[1])

        # Get player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        background_cluster = max(set(corner_clusters), key = corner_clusters.count)
        player_cluster = 1 - background_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        
        # Add player colors to list
        for track_id, player_detection in player_detections.items():
            bounding_box = player_detection['bounding_box']
            player_color = self.get_player_color(frame, bounding_box)
            player_colors.append(player_color)

        # Run KMeans clustering on list of player colors
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', n_init = 10).fit(player_colors)

        self.kmeans = kmeans

        # Assign team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bounding_box, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bounding_box)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1   # No idea what this means
        #team_id += 1

        self.player_team_dict['player_id'] = team_id

        return team_id

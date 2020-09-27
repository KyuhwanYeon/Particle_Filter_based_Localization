/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = 100;
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  particles.resize(num_particles);
  weights.resize(num_particles);
  for (int i = 0; i < num_particles; ++i)
  {
    Particle particle;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    particles[i] = particle;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  // noise for modeling error
  std::default_random_engine gen;
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);

  // to avoid divide by 0
  if (fabs(yaw_rate) < 0.0001)
  {
    yaw_rate = 0.0001;
  }
  for (int i = 0; i < num_particles; ++i)
  {
    double pre_x = particles[i].x;
    double pre_y = particles[i].y;
    double pre_theta = particles[i].theta;
    particles[i].x = pre_x + velocity / yaw_rate * (sin(pre_theta + yaw_rate * delta_t) - sin(pre_theta));
    particles[i].y = pre_y + velocity / yaw_rate * (cos(pre_theta) - cos(pre_theta + yaw_rate * delta_t));
    particles[i].theta = pre_theta + yaw_rate * delta_t;
    // add noise
    particles[i].x += noise_x(gen);
    particles[i].y += noise_y(gen);
    particles[i].theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  std::vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
  double x, y, theta, weight;
  for (int i = 0; i < num_particles; i++)
  {
    Particle particle = particles[i];
    vector<LandmarkObs> close_landmark_list;
    // find the close landmarks (which is in sensor range)
    for (int j = 0; j < landmark_list.size(); j++)
    {
      Map::single_landmark_s landmark = landmark_list[j];
      double dist_landmark2particle = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      if (dist_landmark2particle < sensor_range)
      {
        LandmarkObs close_landmark;
        close_landmark.id = landmark.id_i;
        close_landmark.x = landmark.x_f;
        close_landmark.y = landmark.y_f;
        close_landmark_list.push_back(close_landmark);
      }
    }
    particles[i].weight = calculate_weight(observations, particle, close_landmark_list, sigma_x, sigma_y);
    weights[i] = particles[i].weight;
    //
  }
}
double ParticleFilter::calculate_weight(const vector<LandmarkObs> &observations, Particle &particle, vector<LandmarkObs> close_landmark_list, double sig_x, double sig_y)
{
  double weight = 1;
  for (int i = 0; i < observations.size(); i++)
  {
    LandmarkObs observed_landmark = observations[i];
    // transform coordinate (local coordinate to global coordinate) - observed landmark
    double x_obs = particle.x + cos(particle.theta) * observed_landmark.x - sin(particle.theta) * observed_landmark.y;
    double y_obs = particle.y + sin(particle.theta) * observed_landmark.x + cos(particle.theta) * observed_landmark.y;
    double dist_min = 10000;
    int closest_id = 0;
    // find most closest landmark
    for (int j = 0; j < close_landmark_list.size(); j++)
    {
      LandmarkObs close_landmark = close_landmark_list[j];
      double dist_obs2close = dist(x_obs, y_obs, close_landmark.x, close_landmark.y);
      if (dist_obs2close < dist_min)
      {
        closest_id = close_landmark.id;
        dist_min = dist_obs2close;
      }
    }
    // get x and y of most closest landmark
    double mu_x, mu_y;
    for (int k = 0; k < close_landmark_list.size(); k++)
    {
      if (close_landmark_list[k].id == closest_id)
      {
        mu_x = close_landmark_list[k].x;
        mu_y = close_landmark_list[k].y;
        break;
      }
    }
    // calculate weight
    weight *= multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
  }
  return weight;
}
void ParticleFilter::resample()
{
  // resampling wheel method
  vector<Particle> new_particles;
  double beta = 0;
  //
  int index = rand() % num_particles;
  double w_max = *max_element(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i)
  {
    beta += (rand() / (RAND_MAX + 1.0)) * (2 * w_max);
    while (weights[index] < beta)
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                                   double mu_x, double mu_y)
{
  // Multivariate -gaussian probability
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2))) + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);

  return weight;
}

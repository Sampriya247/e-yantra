import sys

# 👇 Replace this path with your actual folder path
sys.path.append(r"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\programming\zmqRemoteApi\clients\python")

from zmqRemoteApi import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

print("✅ Connected to CoppeliaSim!")
print("Simulation time:", sim.getSimulationTime())

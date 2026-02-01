resource "vkcs_kubernetes_cluster" "credit_scoring" {
  name        = "credit-scoring-${var.environment}"
  cluster_type = "standard"
  network_id   = var.network_id
  
  master_flavor_id = var.master_flavor_id
  master_count     = 1
  
  availability_zone = var.zone
  
  labels = {
    environment = var.environment
    project     = "credit-scoring"
  }
  
  keypair = var.keypair_name
}

resource "vkcs_kubernetes_node_group" "cpu_nodes" {
  cluster_id = vkcs_kubernetes_cluster.credit_scoring.id
  name       = "cpu-nodes-${var.environment}"
  
  node_count = var.cpu_node_count
  flavor_id  = var.cpu_flavor_id
  
  autoscaling {
    min_replicas = var.cpu_min_nodes
    max_replicas = var.cpu_max_nodes
    enabled      = true
  }
  
  labels = {
    "node.kubernetes.io/role" = "worker"
    "accelerator"              = "cpu"
  }
  
  taints = []
}

resource "vkcs_kubernetes_node_group" "gpu_nodes" {
  count = var.enable_gpu ? 1 : 0
  
  cluster_id = vkcs_kubernetes_cluster.credit_scoring.id
  name       = "gpu-nodes-${var.environment}"
  
  node_count = var.gpu_node_count
  flavor_id  = var.gpu_flavor_id
  
  autoscaling {
    min_replicas = var.gpu_min_nodes
    max_replicas = var.gpu_max_nodes
    enabled      = true
  }
  
  labels = {
    "node.kubernetes.io/role" = "worker"
    "accelerator"              = "nvidia-gpu"
  }
  
  taints = [{
    key    = "nvidia.com/gpu"
    value  = "present"
    effect = "NO_SCHEDULE"
  }]
}
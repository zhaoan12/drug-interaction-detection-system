terraform {
  required_version = ">= 1.6.0"
}

variable "project_name" {
  type    = string
  default = "ddi-research-system"
}

resource "aws_s3_bucket" "reference_store" {
  bucket = "${var.project_name}-reference-store"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/ecs/${var.project_name}"
  retention_in_days = 14
}


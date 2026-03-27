output "reference_store_bucket" {
  value = aws_s3_bucket.reference_store.bucket
}

output "api_log_group" {
  value = aws_cloudwatch_log_group.api.name
}

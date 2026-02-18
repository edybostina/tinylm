#pragma once

#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h"
#include "coordinator.h"
#include "../model/tokenizer.h"

class InferenceServiceImpl final : public tinylm::InferenceService::Service {
   private:
	std::shared_ptr<Coordinator> coordinator_;
	Tokenizer &tokenizer_;

   public:
	InferenceServiceImpl(std::shared_ptr<Coordinator> coordinator, Tokenizer &tokenizer);

	grpc::Status GenerateStream(grpc::ServerContext *context, const tinylm::GenerateRequest *request,
								grpc::ServerWriter<tinylm::GenerateResponse> *writer) override;
};

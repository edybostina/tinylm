#include "service.h"
#include "../logger.h"

InferenceServiceImpl::InferenceServiceImpl(std::shared_ptr<Coordinator> coordinator, Tokenizer &tokenizer)
	: coordinator_(coordinator), tokenizer_(tokenizer) {}

grpc::Status InferenceServiceImpl::GenerateStream(grpc::ServerContext *context, const tinylm::GenerateRequest *request,
												  grpc::ServerWriter<tinylm::GenerateResponse> *writer) {
	uint64_t req_id = coordinator_->next_id();
	std::vector<int> token_ids = tokenizer_.encode(request->prompt(), true);

	LOG_INFO("gRPC") << "Received request " << req_id;

	auto my_queue = coordinator_->register_queue(req_id);

	coordinator_->request_queue.push({req_id, request->prompt(), token_ids, request->max_tokens(),
									  request->temperature(), request->top_k(), request->top_p()});

	std::vector<int> gen_ids;
	std::string last_decoded;

	while (true) {
		if (context->IsCancelled()) {
			LOG_INFO("gRPC") << "Client cancelled request " << req_id;
			coordinator_->cancel_request(req_id);
			my_queue->shutdown();
			break;
		}

		StepResult result = my_queue->wait_and_pop().value_or(StepResult{req_id, 0, true});

		gen_ids.push_back(result.new_token_id);
		std::string current_decoded = tokenizer_.decode(gen_ids);
		std::string delta = current_decoded.substr(last_decoded.size());
		last_decoded = std::move(current_decoded);

		tinylm::GenerateResponse response;
		response.set_text(delta);
		response.set_is_finished(result.is_finished);

		if (!writer->Write(response)) {
			break;
		}

		if (result.is_finished) {
			break;
		}
	}

	coordinator_->remove_queue(req_id);
	LOG_INFO("gRPC") << "Finished request " << req_id;

	return grpc::Status::OK;
}

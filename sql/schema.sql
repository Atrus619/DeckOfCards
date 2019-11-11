DROP TABLE if exists cards.experience;

DROP TABLE if exists cards.experience_archive;

DROP TABLE if exists cards.metrics;

DROP SCHEMA if exists cards;

CREATE SCHEMA cards AUTHORIZATION postgres;

CREATE TABLE cards.experience (
    id serial NOT NULL,
    ins_ts timestamp NOT NULL DEFAULT now(),
    agent_id varchar NULL,
    opponent_id varchar NULL,
    run_id varchar NOT NULL,
    vector varchar NOT NULL,
    next_vector varchar NULL,
    reward numeric NULL,
    action varchar NULL,
    CONSTRAINT experience_pk PRIMARY KEY (id)
);

CREATE TABLE cards.metrics (
	id bigserial NOT NULL,
	run_id varchar NOT NULL,
	win_rate numeric NOT NULL,
	win_rate_random numeric NOT NULL,
	win_rate_expert_policy numeric NOT NULL,
	average_reward numeric NOT NULL,
	ins_ts timestamp NOT NULL DEFAULT now(),
	CONSTRAINT win_rate_pk PRIMARY KEY (id)
);


CREATE TABLE cards.experience_archive (
	id numeric NULL,
	ins_ts timestamp NULL,
	agent_id varchar NULL,
	run_id varchar NULL,
	vector varchar NULL,
	reward numeric NULL,
	"action" varchar NULL,
	next_vector varchar NULL
);

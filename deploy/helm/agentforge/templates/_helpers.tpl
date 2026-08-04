{{/* Naming */}}
{{- define "agentforge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentforge.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "agentforge.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Labels */}}
{{- define "agentforge.labels" -}}
helm.sh/chart: {{ include "agentforge.chart" . }}
app.kubernetes.io/name: {{ include "agentforge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: agentforge
{{- end -}}

{{/* Per-component selector. Call as (dict "ctx" . "component" "api"). */}}
{{- define "agentforge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentforge.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "agentforge.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "agentforge.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Image reference. Call as (dict "ctx" . "image" .Values.api.image). */}}
{{- define "agentforge.image" -}}
{{- $registry := .ctx.Values.image.registry -}}
{{- $tag := default .ctx.Chart.AppVersion .image.tag -}}
{{- if $registry -}}
{{- printf "%s/%s:%s" $registry .image.repository $tag -}}
{{- else -}}
{{- printf "%s:%s" .image.repository $tag -}}
{{- end -}}
{{- end -}}

{{/*
Config + secret env for every app pod. envFrom rather than per-key `env` so a
new setting is a values change, not a template change -- and so no secret value
is ever named in a manifest.
*/}}
{{- define "agentforge.envFrom" -}}
- configMapRef:
    name: {{ include "agentforge.fullname" . }}-config
- secretRef:
    name: {{ .Values.secrets.existingSecret }}
    optional: false
{{- end -}}

{{/*
Datastore and tracing endpoints the chart can derive without touching a
credential. POSTGRES_DSN and REDIS_URL are deliberately absent: both embed a
password, so they come from the Secret in full even when the datastore is
in-cluster.
*/}}
{{- define "agentforge.derivedEnv" -}}
{{- if .Values.qdrant.enabled }}
- name: QDRANT_URL
  value: http://{{ include "agentforge.fullname" . }}-qdrant:{{ .Values.qdrant.port }}
{{- end }}
{{- if .Values.tracing.enabled }}
- name: OTEL_ENABLED
  value: "true"
- name: OTEL_SERVICE_NAME
  value: {{ .Values.tracing.serviceName | quote }}
- name: LANGSMITH_PROJECT
  value: {{ .Values.tracing.langsmithProject | quote }}
{{- with .Values.tracing.langsmithEndpoint }}
- name: LANGSMITH_OTEL_ENDPOINT
  value: {{ . | quote }}
{{- end }}
{{- with .Values.tracing.langwatchEndpoint }}
- name: LANGWATCH_OTEL_ENDPOINT
  value: {{ . | quote }}
{{- end }}
{{- with .Values.tracing.arizeEndpoint }}
- name: ARIZE_OTEL_ENDPOINT
  value: {{ . | quote }}
{{- end }}
{{- with .Values.tracing.otlpEndpoint }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ . | quote }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
readOnlyRootFilesystem is on, so anything that writes needs an emptyDir.
Python wants a writable /tmp; that is the whole list.
*/}}
{{- define "agentforge.tmpVolume" -}}
- name: tmp
  emptyDir: {}
{{- end -}}

{{- define "agentforge.tmpMount" -}}
- name: tmp
  mountPath: /tmp
{{- end -}}

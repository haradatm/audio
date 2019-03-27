var localStream = null;
var connection = null;

var disconnect = function () {
	if (localStream) for (track of localStream.getTracks()) track.stop();
	localStream = null;
	if (connection != null) connection.close();
	connection = null;
};

var connect = function () {
	// WebSocketのコネクションを作成する
	connection = new WebSocket('ws://localhost:8888/ser');

	// ブラウザ間の表記ゆれを吸収する
	window.AudioContext = window.AudioContext || window.webkitAudioContext;

	// AudioContext を作成する
	var audioContext = new AudioContext();

	// 音声処理初期設定 (getUserMedia のコールバックとして,マイクとの接続開始時に実行)
	function startUserMedia(stream) {
		localStream = stream;
		var voice = new Float32Array(0);

		// 音声ストリーム音源オブジェクトの作成
		var source = audioContext.createMediaStreamSource(stream);

		// VAD のオプション設定
		var options = {

			// 区間検出対象となるストリーム音源オブジェクトの指定
			source: source,

			// WebSocketのコネクションの指定
			connection: connection,

			// 音声区間検出終了時ハンドラ指定
			voice_start: function () {
				console.log('voice start');
				voice = new Float32Array(0);
			},

			// 音声区間検出開始時ハンドラ指定
			voice_stop: function () {
				console.log('voice stop');
				if (connection != null && connection.readyState === WebSocket.OPEN) {
					connection.send(voice);
				}
			},

			// 音声受信時のハンドラ指定
			callback: function (e) {
				var input = e.inputBuffer.getChannelData(0);
				var buffer = new Float32Array(voice.length + input.length);
				for (var i = 0; i < voice.length; i++) {
					buffer[i] = voice[i];
				}
				for (var i = 0; i < input.length; i++) {
					buffer[voice.length + i] = input[i];
				}
				voice = buffer;
			},

		};

		// VADオブジェクト作成 (以降使用しない)
		var vad = new VAD(options);
	};

	// ブラウザ間の表記ゆれを吸収する
	navigator.getUserMedia = navigator.getUserMedia || navigator.mozGetUserMedia || navigator.webkitGetUserMedia;

	// getUserMedia を起動し,マイクアクセスを開始する
	navigator.getUserMedia(
		// オプション
		{audio: true, video: false},

		// 成功時コールバック
		startUserMedia,

		// エラー時コールバック
		function (e) {
			console.log("No live audio input in this browser: " + e);
		}
	);
};
